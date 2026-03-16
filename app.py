import selectors
import gradio as gr
import xml.etree.ElementTree as ET
import json
import os
import torch
import nest_asyncio
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from collections import defaultdict
from pyvis.network import Network
import tempfile
import re
import html
import time

# ===================== CONFIGURATION =====================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Mapping logic for OPC UA to AAS standards
OPCUA_TO_AAS_MAP = {
    "attributes": {
        "NodeId": "id",
        "BrowseName": "idShort",
        "DisplayName": "idShort",
        "Description": "description",
        "DataType": "valueType",
        "Value": "value",
        "EngineeringUnits": "unit",
        "EURange": "Range",
        "NodeVersion": "administration.version"
    },
    "node_classes": {
        "UAObject": "SubmodelElementCollection",
        "UAVariable": "Property",
        "ObjectType": "Submodel",
        "ReferenceType": None
    },
    "data_types": {
        "Boolean": "xs:boolean",
        "SByte": "xs:int",
        "Byte": "xs:byte",
        "Int16": "xs:int",
        "UInt16": "xs:int",
        "Int32": "xs:int",
        "UInt32": "xs:int",
        "Int64": "xs:int",
        "UInt64": "xs:int",
        "Float": "xs:float",
        "Double": "xs:double",
        "String": "xs:string",
        "DateTime": "xs:dateTime",
        "ByteString": "xs:base64Binary",
        "LocalizedText": "text",
    },
    "reference_types": {
        "HasComponent": "SubmodelElements",
        "HasProperty": "SubmodelElements",
        "Organizes": "SubmodelElements",
        "HasTypeDefinition": "semanticId"
    }
}

# ===================== SILENCE PATCH =====================
_original_fileobj_to_fd = selectors._fileobj_to_fd
def _quiet_fileobj_to_fd(fileobj):
    try: return _original_fileobj_to_fd(fileobj)
    except ValueError: return -1
selectors._fileobj_to_fd = _quiet_fileobj_to_fd
nest_asyncio.apply()

# ===================== HELPER FUNCTIONS =====================

def clean_aas_json(data):
    """Recursively removes nulls and N/A values from the dictionary for a cleaner AAS file."""
    if isinstance(data, dict):
        for key in list(data.keys()):
            val = data[key]
            if val is None or val == "N/A" or val == "":
                del data[key]
            else:
                clean_aas_json(val)
    elif isinstance(data, list):
        for item in data:
            clean_aas_json(item)
    return data

def extract_json_from_llm(text: str):
    """Safely extract the first JSON object from LLM output."""
    if not text:
        return None
    # Remove markdown fences
    text = re.sub(r"```json|```", "", text).strip()
    
    # Attempt to find JSON block
    start = text.find("{")
    end = text.rfind("}")
    
    # If no curly brace, check for array
    if start == -1 or end == -1:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None
    
    json_candidate = text[start:end + 1]
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return None

# ===================== 1. GPU SEARCH ENGINE =====================
class GPUSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing SentenceTransformer on: {self.device.upper()}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus_embeddings = None
        self.chunks = [] 

    def index_data(self, chunks):
        self.chunks = chunks
        texts = [c['content'] for c in chunks]
        self.corpus_embeddings = self.model.encode(texts, convert_to_tensor=True)
        return f"✅ Indexed {len(texts)} nodes on {self.device.upper()}"

    def search(self, query, top_k=10):
        exact_matches = []
        id_pattern = re.search(r'((?:ns=\d+;)?i=\d+|ns=\d+;[sbg]=[^"\s]+)', query)
        if id_pattern:
            target_id = id_pattern.group(1)
            for chunk in self.chunks:
                if chunk['metadata'].get('NodeId') == target_id:
                    exact_matches.append({'chunk': chunk, 'score': 2.0}) 
                elif target_id in chunk['metadata'].get('NodeId', ''):
                    exact_matches.append({'chunk': chunk, 'score': 1.5}) 

        if self.corpus_embeddings is None: return []
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        k = min(top_k, len(self.chunks))
        top_results = torch.topk(scores, k=k)
        
        vector_matches = []
        for score, idx in zip(top_results.values, top_results.indices):
            vector_matches.append({'chunk': self.chunks[idx.item()], 'score': score.item()})

        seen_ids = set()
        final_results = []
        for res in exact_matches + vector_matches:
            nid = res['chunk']['metadata'].get('NodeId')
            if nid not in seen_ids:
                final_results.append(res)
                seen_ids.add(nid)
        return final_results[:top_k]

# ===================== 2. XML PARSER =====================

def build_tree_xml_chunks(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        main_ns = root.tag.split('}')[0] + "}" if '}' in root.tag else ""
        
        def get_deep_text(elem):
            if elem.text and elem.text.strip(): return elem.text.strip()
            joined = " ".join([t.strip() for t in elem.itertext() if t.strip()])
            return joined if joined else "N/A"

        nodes, children_map, potential_roots = {}, defaultdict(list), set()
        all_elements = root.findall(f".//{main_ns}*")
        
        for elem in all_elements:
            nid = elem.attrib.get("NodeId", "").strip()
            if not nid: continue
            data = {k: v for k, v in elem.attrib.items()}
            data['Tag'] = elem.tag.replace(main_ns, "")
            for child in elem:
                if "References" not in child.tag:
                    data[child.tag.replace(main_ns, "")] = get_deep_text(child)
            
            refs = elem.find(f"{main_ns}References")
            if refs is not None:
                all_refs = []
                for ref in refs:
                    all_refs.append({
                        'TargetId': ref.text.strip(), 
                        'Type': ref.attrib.get("ReferenceType", ""), 
                        'IsForward': ref.attrib.get("IsForward", "true").lower() == "true"
                    })
                    if all_refs[-1]['IsForward'] and any(x in all_refs[-1]['Type'] for x in ["Component", "Organizes", "Property"]):
                        children_map[nid].append(all_refs[-1]['TargetId'])
                        potential_roots.discard(all_refs[-1]['TargetId'])
                data['AllReferences'] = all_refs
            nodes[nid] = data
            potential_roots.add(nid)

        chunks, visited = [], set()
        def traverse(node_id, ancestors):
            if node_id in visited or node_id not in nodes: return
            visited.add(node_id)
            curr = nodes[node_id]
            name = curr.get('DisplayName', curr.get('BrowseName', 'Unnamed'))
            chunks.append({
                'content': f"Node: {name}\nID: {node_id}\nType: {curr.get('Tag')}", 
                'metadata': {
                    'NodeId': node_id, 
                    'Name': name, 
                    'Lineage': ancestors + [curr], 
                    'AllReferences': curr.get('AllReferences', [])
                }
            })
            for cid in children_map[node_id]: traverse(cid, ancestors + [curr])

        for root_id in list(potential_roots): traverse(root_id, [])
        return chunks
    except Exception as e: raise ValueError(f"XML Error: {str(e)}")

# ===================== 3. CONVERSION ENGINES =====================

def convert_single_node_to_aas(selected_label, candidate_storage):
    """Converts a specific selected node from search results to AAS snippet within standard structure."""
    if not selected_label or not candidate_storage:
        return "⚠️ Please select a node from the search results first."
    
    try:
        idx = int(selected_label.split('.')[0]) - 1
        node_meta = candidate_storage[idx]['chunk']['metadata']
        
        # Prepare context for the node
        target_node = node_meta['Lineage'][-1]
        node_data = {
            "opc_name": node_meta['Name'],
            "opc_id": node_meta['NodeId'],
            "opc_type": target_node.get('Tag', 'Unknown'),
            "opc_value": target_node.get('Value', target_node.get('text', 'N/A'))
        }

        # Enhanced prompt for strict AAS compliance and key mapping
        prompt = f"""
Translate the following OPC UA node data into a standard AAS SubmodelElement.

### Mapping Rules:
1. Map 'opc_id' to the AAS 'id' field.
2. Map 'opc_name' (even if it's a URL) directly to the AAS 'idShort' field.
3. Use 'opc_type' to determine the AAS 'modelType' (e.g., UAObject -> SubmodelElementCollection, UAVariable -> Property).
4. Full Attribute Rules: {json.dumps(OPCUA_TO_AAS_MAP, indent=2)}

### Input Node Data:
{json.dumps(node_data, indent=2)}

### Output Requirements:
Return ONLY a valid JSON list of objects for the 'submodelElements' field.
Ensure the structure is standard: {{"modelType": {{"name": "..."}}, "idShort": "...", "id": "..."}}
"""
        with Groq(api_key=GROQ_API_KEY) as client:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            resp_raw = resp.choices[0].message.content.strip()
            parsed = extract_json_from_llm(resp_raw)
            
            if parsed:
                # Handle dictionary response wrapping the list
                elements = parsed if isinstance(parsed, list) else parsed.get("submodelElements", [parsed])
                
                # Wrap in the requested standard structure
                standard_structure = {
                    "assetAdministrationShells": [
                        {
                            "idShort": "FullExport",
                            "submodels": [
                                {
                                    "keys": [
                                        {
                                            "value": "MainSubmodel"
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "submodels": [
                        {
                            "idShort": "MainSubmodel",
                            "modelType": {"name": "Submodel"},
                            "submodelElements": elements
                        }
                    ]
                }
                
                clean_result = clean_aas_json(standard_structure)
                return json.dumps(clean_result, indent=4)
            return f"❌ Failed to generate valid JSON. Raw response:\n{resp_raw}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_full_xml_iterative(file_obj, progress=gr.Progress(track_tqdm=True)):
    if not file_obj: return "⚠️ No file uploaded.", gr.update(visible=False)
    
    print("\n🚀 STARTING FULL XML TO AAS CONVERSION")
    try:
        tree = ET.parse(file_obj.name)
        root = tree.getroot()
        ns = root.tag.split('}')[0] + "}" if '}' in root.tag else ""
        
        all_nodes = root.findall(f".//{ns}UAObject") + root.findall(f".//{ns}UAVariable")
        extracted_data = []
        for node in all_nodes:
            dn = node.find(f"{ns}DisplayName")
            name = dn.text if dn is not None else node.get("BrowseName", "Unnamed")
            val_elem = node.find(f".//{ns}Value")
            value = "".join(val_elem.itertext()).strip() if val_elem is not None else "N/A"
            extracted_data.append({
                "opc_name": name,
                "opc_id": node.get("NodeId", "Unknown"),
                "opc_type": node.tag.replace(ns, ""),
                "opc_value": value
            })

        master_elements = []
        batch_size = 5 
        total_batches = (len(extracted_data) + batch_size - 1) // batch_size

        with Groq(api_key=GROQ_API_KEY) as client:
            for i in range(0, len(extracted_data), batch_size):
                progress(i / len(extracted_data), desc=f"Batch {i//batch_size+1}/{total_batches}")
                batch = extracted_data[i:i + batch_size]
                
                prompt = f"""
Translate the following OPC UA nodes to AAS SubmodelElements.

### Instructions:
1. Map 'opc_id' to AAS 'id'.
2. Map 'opc_name' to AAS 'idShort'.
3. Follow these class and type rules: {json.dumps(OPCUA_TO_AAS_MAP, indent=2)}

### Input Data:
{json.dumps(batch, indent=2)}

Return STRICT JSON ONLY in the format:
{{ "submodelElements":[ ... ] }}
Do not include explanations or markdown. Ensure correct modelType structure: {{"modelType": {{"name": "..."}}}}
"""
                retries = 2
                success = False
                while retries >= 0 and not success:
                    try:
                        resp = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0
                        )
                        resp_raw = resp.choices[0].message.content.strip()
                        parsed = extract_json_from_llm(resp_raw)
                        if parsed is None: raise ValueError("Invalid JSON from LLM")
                        elements = parsed.get("submodelElements", []) if isinstance(parsed, dict) else parsed
                        master_elements.extend([el for el in elements if isinstance(el, dict) and ("id" in el or "idShort" in el)])
                        success = True
                    except Exception as e:
                        print(f"⚠️ Batch {i} failed: {e}")
                        retries -= 1
                        time.sleep(0.5)

        final_aas = {
            "assetAdministrationShells": [{"idShort": "FullExport", "submodels": [{"keys": [{"value": "MainSubmodel"}]}]}],
            "submodels": [{
                "idShort": "MainSubmodel",
                "modelType": {"name": "Submodel"},
                "submodelElements": master_elements 
            }]
        }
        
        final_aas = clean_aas_json(final_aas)
        final_path = os.path.join(tempfile.gettempdir(), "complete_aas_export.json")
        with open(final_path, "w", encoding='utf-8') as f:
            json.dump(final_aas, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        preview = json.dumps(final_aas, indent=2)[:20000] + "\n\n... preview truncated ..."
        return preview, gr.update(value=final_path, visible=True)

    except Exception as e:
        return f"❌ Conversion Error: {str(e)}", gr.update(visible=False)

# ===================== 4. VISUALIZATION =====================

def generate_interactive_graph(selected_label, candidate_storage):
    if not selected_label or not candidate_storage:
        return "<div>Select a candidate to visualize the graph.</div>"

    try:
        # Extract index from label "1. NodeName (Score: 0.99)"
        idx = int(selected_label.split('.')[0]) - 1
        chunk_data = candidate_storage[idx]['chunk']
        metadata = chunk_data.get("metadata", {})
        lineage = metadata.get("Lineage", [])
        all_refs = metadata.get("AllReferences", [])

        if not lineage:
            return "<div>No lineage available for visualization.</div>"

        target_node = lineage[-1]
        def get_id(node): return node.get("NodeId") or node.get("id") or node.get("Name")
        def get_label(node): return node.get("DisplayName") or node.get("Name") or node.get("idShort") or "Node"
        
        target_id = get_id(target_node)
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True, cdn_resources="in_line")
        added_nodes = set()

        def safe_add_node(nid, label, color, title, size=25):
            if nid and nid not in added_nodes:
                display_label = label[:18] + "..." if len(label) > 18 else label
                net.add_node(nid, label=display_label, title=title, color=color, size=size)
                added_nodes.add(nid)

        # Draw hierarchy
        for i, node in enumerate(lineage):
            nid = get_id(node)
            label = get_label(node)
            safe_add_node(nid, label, "#4CAF50" if i == len(lineage)-1 else "#2196F3", f"ID: {nid}", size=40 if i == len(lineage)-1 else 25)
            if i > 0:
                net.add_edge(get_id(lineage[i-1]), nid, color="#B0BEC5")

        # Draw references
        for ref in all_refs:
            ref_id = ref.get("TargetId")
            ref_type = str(ref.get("Type", "Reference"))
            is_fwd = ref.get("IsForward", True)
            if not ref_id: continue
            
            color = "#9C27B0" if "TypeDefinition" in ref_type else ("#FFC107" if is_fwd else "#9E9E9E")
            safe_add_node(ref_id, ref_id[:12], color, f"ID: {ref_id}\nType: {ref_type}", size=20)
            if is_fwd:
                net.add_edge(target_id, ref_id, title=ref_type, color=color)
            else:
                net.add_edge(ref_id, target_id, title=f"{ref_type} (Reverse)", color=color, dashes=True)

        net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -4000, "springLength": 120}}, "interaction": {"hover": true}}')
        tmp_path = os.path.join(tempfile.gettempdir(), "graph.html")
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f: raw_html = f.read()
        return f"<iframe style='width:100%; height:600px; border:none;' srcdoc='{html.escape(raw_html)}'></iframe>"

    except Exception as e:
        return f"<div style='color:red'>Graph error: {str(e)}</div>"

# ===================== 5. UI LOGIC =====================

def process_file_and_index(file_obj):
    if not file_obj: return "⚠️ No File", None
    try:
        chunks = build_tree_xml_chunks(file_obj.name)
        engine = GPUSearchEngine()
        status = engine.index_data(chunks)
        return status, engine
    except Exception as e: return f"❌ Error: {str(e)}", None

def run_query_and_populate(query, engine_state):
    if not engine_state: return "⚠️ Index First", "", gr.update(choices=[]), []
    top_results = engine_state.search(query, top_k=10)
    if not top_results: return "No results found.", "", gr.update(choices=[]), []

    choices = [f"{i+1}. {res['chunk']['metadata']['Name']} (Score: {res['score']:.2f})" for i, res in enumerate(top_results)]
    trace_log = "\n\n".join([f"--- Match {i+1} ---\n{res['chunk']['content']}" for i, res in enumerate(top_results)])
    
    try:
        with Groq(api_key=GROQ_API_KEY) as client:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": f"Summarize based on data: {top_results[0]['chunk']['content']}. User asked: {query}"}],
                max_tokens=200
            ).choices[0].message.content.strip()
            answer = resp
    except: answer = "Search complete. Inspect the graph and traces for details."

    return answer, trace_log, gr.update(choices=choices, value=choices[0]), top_results

# ===================== UI LAYOUT =====================

with gr.Blocks(title="Industrial Visualizer") as demo:
    gr.Markdown("# 🏭 Industrial GraphRAG + AAS Generator")
    engine_state = gr.State(None)
    candidates_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("🔍 Search & Analyze"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Upload OPC UA XML")
                    idx_btn = gr.Button("1. Index Data", variant="secondary")
                    status = gr.Textbox(label="Index Status")
                with gr.Column(scale=2):
                    q_in = gr.Textbox(label="Query Nodes or Properties")
                    ask_btn = gr.Button("2. Search & Analyze", variant="primary")
                    ans_out = gr.Markdown(label="Verified Answer")
            
            with gr.Row():
                with gr.Column(scale=2):
                    vis_dropdown = gr.Dropdown(label="Visualize Result", choices=[])
                    html_vis = gr.HTML()
                    
                    gr.Markdown("---")
                    gr.Markdown("### ⚙️ Node-to-AAS Converter")
                    single_aas_btn = gr.Button("Convert Selected Node to AAS Snippet", variant="secondary")
                    single_aas_out = gr.Code(language="json", label="AAS Snippet")
                    
                with gr.Column(scale=1):
                    with gr.Accordion("Raw Node Data", open=True):
                        trace_out = gr.Textbox(lines=15, label="Metadata")

        with gr.Tab("📦 Full File Conversion"):
            gr.Markdown("### 🚀 Iterative Mass Conversion to AAS")
            full_conv_btn = gr.Button("Start Full XML → AAS Translation", variant="primary")
            with gr.Row():
                full_download = gr.DownloadButton("Download Complete AAS JSON", visible=False)
            full_preview = gr.Code(language="json", label="JSON Preview")

    # Event Connections
    idx_btn.click(process_file_and_index, inputs=[file_in], outputs=[status, engine_state])
    ask_btn.click(run_query_and_populate, inputs=[q_in, engine_state], outputs=[ans_out, trace_out, vis_dropdown, candidates_state])
    
    vis_dropdown.change(generate_interactive_graph, inputs=[vis_dropdown, candidates_state], outputs=[html_vis])
    single_aas_btn.click(convert_single_node_to_aas, inputs=[vis_dropdown, candidates_state], outputs=[single_aas_out])
    
    full_conv_btn.click(process_full_xml_iterative, inputs=[file_in], outputs=[full_preview, full_download])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())