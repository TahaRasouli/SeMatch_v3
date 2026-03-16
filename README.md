# 🏭 ISW's Industrial OPC UA + AAS Intelligence

An advanced industrial data processing tool designed to bridge the gap between **OPC UA (Information Models)** and **AAS (Asset Administration Shell)** standards.  

This application leverages **GraphRAG (Retrieval-Augmented Generation over Graphs)** and **LLM-driven iterative processing** to analyze, visualize, and convert complex industrial **XML/JSON models**.

---

# 🚀 Key Features

## 1. GraphRAG Search & Analysis

**Semantic Indexing**  
Uses `SentenceTransformers` and **GPU acceleration (where available)** to index industrial nodes based on their **BrowseNames**, **DisplayNames**, and **hierarchical context**.

**Contextual Retrieval**  
Performs **vector similarity search** combined with **exact NodeId matching** to locate specific components within massive industrial datasets.

**Verified Reasoning**  
Uses **LLMs (via Groq)** to analyze retrieved nodes and provide **human-readable answers** about the model's structure and properties.

---

## 2. Interactive Visualization

**Hierarchical Graphing**  
Dynamically generates **interactive 2D graphs** using `PyVis` to show a node's:

- Ancestors
- Siblings
- Forward / Reverse references

**Color-Coded Relationships**

- 🟢 **Target Node** – The central element of your query  
- 🔵 **Hierarchy (Lineage)** – The path from the root to the target  
- 🟣 **Type Definitions** – Links to formal OPC UA Type definitions  
- 🟡 **Forward References** – Components or properties owned by the node  

---

## 3. Iterative AAS Conversion

**Batch Processing**  
Handles **large XML files** by processing nodes in **iterative batches**, preventing **context window overflows** and ensuring system stability.

**Standard Alignment**  
Automatically maps OPC UA attributes to **AAS structures**:

- `BrowseName → idShort`
- `UAObject → SubmodelElementCollection`

**Post-Conversion Polishing**

Includes a recursive **Cleaning Engine** that:

- removes `null` values  
- removes `N/A` fields  
- produces a **production-ready, standards-compliant JSON file**

**Single Node Converter**

Ability to convert a **single discovered node** into a full **AAS JSON structure** directly from the search interface.

---

# 🛠️ Technical Stack

**Core**

- Python **3.9+**

**AI / LLM**

- Groq API (`openai/gpt-oss-20b`)
- SentenceTransformers (`all-MiniLM-L6-v2`)

**UI Framework**

- Gradio (Modern **6.0+** layout)

**Data Parsing**

- `xml.etree.ElementTree`
- `json`

**Visualization**

- `pyvis` (Interactive HTML graphs)

**Async Handling**

- `nest_asyncio`
- `selectors`

---

# ⚙️ Configuration

The system uses a predefined **mapping dictionary** (`OPCUA_TO_AAS_MAP`) to translate between industrial protocols.

| OPC UA Attribute | AAS Property | Mapping Logic |
|------------------|-------------|---------------|
| NodeId | id | Unique Identifier |
| BrowseName | idShort | Technical Identifier |
| UAVariable | Property | Data point |
| UAObject | SubmodelElementCollection | Structural Grouping |
| DataType | valueType | XML Schema (`xs:`) types |

---

# 📦 Setup & Installation

## 1️⃣ Install Dependencies

```bash
pip install gradio sentence-transformers torch groq pyvis nest_asyncio
```

---

## 2️⃣ API Key Setup

Ensure your **Groq API key** is set inside `industrial_visualizer.py`:

```python
GROQ_API_KEY = "your_gsk_key_here"
```

---

## 3️⃣ Run the Application

```bash
python industrial_visualizer.py
```

---

# 📖 Usage Guide

## Step 1: Data Ingestion

Drop an **OPC UA XML NodeSet file** into the upload box.

Click:

```
1. Index Data
```

This prepares the **GraphRAG engine**.

---

## Step 2: Discovery

Use the **Search & Analyze** tab to query specific parts of your machine model.

Example queries:

```
ns=1;i=5002
```

or

```
Find the identification submodel
```

---

## Step 3: Visualization & Snippets

Select a search result from the dropdown to:

- View the **interactive relationship graph**
- Explore node hierarchy and references

Use the **Node-to-AAS Converter** below the graph to generate a **specific AAS JSON snippet** for that node.

---

## Step 4: Full Conversion

Switch to the **Full File Conversion** tab.

The system will:

1. Process nodes in batches
2. Generate a complete **AAS JSON file**
3. Clean the output structure

Once the progress bar reaches **100%**, download the final JSON.

---

# 📄 License

This project is developed for **industrial modeling and interoperability research**.

Created with **Industrial Visualizer Engine**.