# Specification: End-to-End Financial Fraud Detection Workshop (Cloud Spanner Graph, BigQuery, GNN & MCP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahurtado/fraud-detection/blob/main/assets/fraud_detection_workshop.ipynb)

## 1. Overview & Objectives
This specification defines the implemented architecture, repository layout, data model, and 11-section interactive workshop workflow for the **Financial Fraud Detection Lab** hosted at [`https://github.com/mahurtado/fraud-detection`](https://github.com/mahurtado/fraud-detection).

The lab guides practitioners through building an end-to-end **Anti-Money Laundering (AML) & Financial Fraud Detection** system on Google Cloud that unifies:
1. **Analytical Staging & Exploration in BigQuery**: Staging synthetic payment network CSVs from Google Cloud Storage (`gs://{BUCKET_NAME}/data/*.csv`) into BigQuery (`{PROJECT_ID}.{BQ_DATASET}`) and analyzing tabular distributions with `%%bqsql` and Matplotlib/Seaborn visualizations.
2. **Operational Property Graph in Cloud Spanner (`FraudGraph`)**: Modeling `Person`, `Account`, and `Device` node tables alongside physically interleaved `Owns`, `UsesDevice`, and `Transfer` edge tables (`assets/schema.sql`).
3. **Bidirectional BigQuery $\leftrightarrow$ Cloud Spanner Interoperability**:
   - **Reverse ETL (`EXPORT DATA`)**: Provisioning a BigQuery `ENTERPRISE` reservation (`--job_type=PIPELINE`) and exporting parent node tables followed by interleaved child edge tables directly from BigQuery into Cloud Spanner (`format='CLOUD_SPANNER'`, `spanner_options='{"table": "<TableName>"}'`).
   - **Zero-ETL Federation (`EXTERNAL_DATASET`)**: Linking live Cloud Spanner tables into BigQuery (`CREATE EXTERNAL SCHEMA ... OPTIONS(external_source='google-cloudspanner:///...')`) for federated cross-table analytics without data duplication.
4. **Interactive Multi-Hop Fraud Pattern Discovery & Student Challenges (`%%spanner_graph`)**: Detecting 6 realistic AML typologies using ISO GQL (`GRAPH FraudGraph MATCH ...`) rendered visually in the notebook via `spanner-graph-notebook` (`%%spanner_graph`), paired with 6 hands-on student challenges (`assets/CHALLENGE_SOLUTIONS.md`).
5. **Predictive Graph ML (2-Hop Heterogeneous GraphSAGE GNN, `COSINE_DISTANCE` & 2D PCA)**: Training a 2-hop GNN (`assets/dgf_gnn_pipeline.py`) that writes `128`-dimensional L2-normalized embeddings back into `Account.embedding` (`ARRAY<FLOAT64>(vector_length=>128)`), querying structural nearest neighbors via `COSINE_DISTANCE`, and projecting embeddings into 2D using `bigframes.ml.decomposition.PCA`.
6. **Agentic Fraud Investigation via Model Context Protocol (`MCP`)**: Exposing `FraudGraph` traversal and GNN similarity tools (`get_fraud_graph_summary`, `detect_aml_pattern`, `investigate_account`, `execute_readonly_gql`) over JSON-RPC 2.0 (`assets/mcp_fraud_server.py`).

---

## 2. Repository Layout (`https://github.com/mahurtado/fraud-detection`)

All workshop code, SQL schemas, notebooks, challenge solutions, and evaluation reports reside in `assets/`, while the deterministic synthetic CSV datasets reside in `data/`:

```text
fraud-detection/
├── README.md                              # Architecture diagrams, Mermaid fraud graphs & quickstart
├── SPEC_FRAUD.md                          # Project technical specification (this file)
├── .gitignore                             # Ignores Python cache, checkpoints, and logs
├── assets/
│   ├── fraud_detection_workshop.ipynb     # 11-section (0..10) interactive Colab/Jupyter notebook (83 cells)
│   ├── CHALLENGE_SOLUTIONS.md             # Reference %%spanner_graph ISO GQL solutions for all 6 challenges
│   ├── schema.sql                         # Cloud Spanner DDL (Node/Edge tables, indexes, FraudGraph)
│   ├── queries_gql.sql                    # Standalone ISO GQL & Hybrid BigQuery-Spanner SQL reference
│   ├── dgf_gnn_pipeline.py                # 2-Hop Heterogeneous GraphSAGE GNN training & Spanner write-back
│   ├── gnn_evaluation_report.json         # Verified GNN evaluation metrics & cosine similarity rankings
│   ├── mcp_fraud_server.py                # Model Context Protocol (MCP) JSON-RPC 2.0 investigation server
│   ├── generate_fraud_data.py             # Deterministic synthetic AML dataset generator (seed=42)
│   └── load_spanner_data.py               # Direct Cloud Spanner REST mutation loader utility
└── data/
    ├── Person.csv                         # 2,000 customer/entity KYC profiles
    ├── Account.csv                        # 2,500 financial accounts (60 fraud across 6 patterns, 2,440 legit)
    ├── Device.csv                         # 1,800 authentication devices (Mobile, Web, API, Emulator)
    ├── Owns.csv                           # 2,500 Person -> Account ownership edges
    ├── UsesDevice.csv                     # 2,518 Account -> Device login session edges
    └── Transfer.csv                       # 15,000 Account -> Account transfer edges (88 flagged AML tx)
```

---

## 3. Configurable Environment Parameters (`# @param`)

The notebook (`assets/fraud_detection_workshop.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahurtado/fraud-detection/blob/main/assets/fraud_detection_workshop.ipynb)) and helper scripts avoid hardcoded project names or user emails. All settings are configured via Colab UI parameters (`# @param`) in Section `0.1`:

| Parameter | Default Placeholder | Description |
| :--- | :--- | :--- |
| `PROJECT_ID` | `"your-project-id"` | Target Google Cloud Project ID. |
| `REGION` | `"europe-west1"` | Regional location for BigQuery datasets, reservations, and Cloud Spanner. |
| `BUCKET_NAME` | `"your-gcs-bucket-name"` | GCS bucket where `data/*.csv` files are staged (`gs://{BUCKET_NAME}/data/`). |
| `GITHUB_REPO` | `"https://github.com/mahurtado/fraud-detection"` | Source GitHub repository cloned/downloaded automatically in Step `0.3`. |
| `GITHUB_BRANCH` | `"main"` | Branch of the GitHub repository used for downloading `assets/*` and `data/*.csv`. |
| `BQ_DATASET` | `"fraud_dataset"` | BigQuery staging dataset holding the 6 raw node and edge tables. |
| `BQ_SPANNER_EXTERNAL_DATASET` | `"fraud_spanner_external"` | Zero-ETL BigQuery external schema linked to live Cloud Spanner. |
| `BQ_RESERVATION` | `"fraud-spanner-export-res"` | BigQuery `ENTERPRISE` reservation (`0` baseline, `50` autoscale max slots) for Reverse ETL `PIPELINE` jobs. |
| `SPANNER_INSTANCE` | `"finance-workshop"` | Cloud Spanner instance (`ENTERPRISE` edition, `100` processing units). |
| `SPANNER_DATABASE` | `"fraud"` | Cloud Spanner database hosting `FraudGraph`. |

---

## 4. Synthetic Dataset & Property Graph Schema (`FraudGraph`)

Generated deterministically by `assets/generate_fraud_data.py` (`--seed 42`) and defined in `assets/schema.sql`:

### 4.1 Node Tables
1. **`Person` (`2,000` rows, `PRIMARY KEY (person_id)`)**:
   - Columns: `person_id INT64`, `full_name STRING(MAX)`, `email STRING(MAX)`, `phone STRING(MAX)`, `address STRING(MAX)`, `country STRING(MAX)`, `kyc_tier STRING(MAX)`, `risk_score FLOAT64`, `is_suspect BOOL`, `create_time TIMESTAMP`.
   - Secondary Indexes: `PersonByEmail (email)`, `PersonByPhone (phone)`.
2. **`Account` (`2,500` rows, `PRIMARY KEY (account_id)`)**:
   - Columns: `account_id INT64`, `account_type STRING(MAX)`, `status STRING(MAX)`, `balance FLOAT64`, `country STRING(MAX)`, `is_fraud INT64` (`60` fraudulent accounts = `2.40%`, `2,440` legitimate), `fraud_pattern STRING(MAX)`, `create_time TIMESTAMP`, `embedding ARRAY<FLOAT64>(vector_length=>128)`.
   - Secondary Index: `AccountByFraudPattern (fraud_pattern, is_fraud)`.
3. **`Device` (`1,800` rows, `PRIMARY KEY (device_id)`)**:
   - Columns: `device_id INT64`, `device_type STRING(MAX)`, `os_version STRING(MAX)`, `fingerprint_hash STRING(MAX)`, `ip_subnet STRING(MAX)`, `is_vpn_or_proxy BOOL`, `risk_score FLOAT64`, `create_time TIMESTAMP`.
   - Secondary Index: `DeviceByFingerprint (fingerprint_hash)`.

### 4.2 Interleaved Edge Tables
1. **`Owns` (`2,500` rows, `PRIMARY KEY (person_id, account_id)`, `INTERLEAVE IN PARENT Person ON DELETE CASCADE`)**:
   - Columns: `person_id INT64`, `account_id INT64`, `role STRING(MAX)`, `create_time TIMESTAMP`.
2. **`UsesDevice` (`2,518` rows, `PRIMARY KEY (account_id, device_id)`, `INTERLEAVE IN PARENT Account ON DELETE CASCADE`)**:
   - Columns: `account_id INT64`, `device_id INT64`, `login_count INT64`, `ip_address STRING(MAX)`, `last_used_time TIMESTAMP`.
3. **`Transfer` (`15,000` rows, `PRIMARY KEY (account_id, to_account_id, transfer_id)`, `INTERLEAVE IN PARENT Account ON DELETE CASCADE`)**:
   - Columns: `transfer_id STRING(36)`, `account_id INT64`, `to_account_id INT64`, `amount FLOAT64`, `currency STRING(MAX)`, `transfer_type STRING(MAX)`, `create_time TIMESTAMP`, `step_num INT64`, `is_flagged_tx BOOL` (`88` flagged AML transfers), `fraud_pattern STRING(MAX)`.
   - Secondary Index: `TransferByRecipientTime (to_account_id, create_time)`.

---

## 5. End-to-End Notebook Structure (`assets/fraud_detection_workshop.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahurtado/fraud-detection/blob/main/assets/fraud_detection_workshop.ipynb)

The notebook is organized into **11 sections (`0` through `10`, `83` cells total)**:

| Section | Title & Implementation Details |
| :--- | :--- |
| **Section 0** | **Initial Configuration Parameters, Enable Required APIs, Authentication & GitHub Download**: Configures `# @param` fields (`0.1`), authenticates and enables `spanner`, `bigquery`, `bigqueryconnection`, `bigqueryreservation`, and `storage` APIs (`0.2`), clones/downloads `assets/*` and `data/*.csv` from `https://github.com/mahurtado/fraud-detection`, syncs `data/*.csv` to `gs://{BUCKET_NAME}/data/`, and loads `%bigframes` (`%%bqsql`) and `%spanner_graphs` (`%%spanner_graph`) (`0.3`). |
| **Section 1** | **BigQuery Dataset Creation, Table Creation & Data Loading from GCS**: Creates `{PROJECT_ID}:{BQ_DATASET}` (`1.1`), loads the 3 Node tables (`1.2`) and 3 Edge tables (`1.3`) from `gs://{BUCKET_NAME}/data/*.csv`, and verifies row counts via `%%bqsql` (`1.4`). |
| **Section 2** | **Explore the Data in BigQuery & Justify the Need for Spanner Graph**: Analyzes `Account` fraud pattern breakdown and `Transfer` AML volume using `%%bqsql` + Matplotlib/Seaborn charts, and contrasts relational multi-join complexity against Spanner Graph interleaved index-free adjacency. |
| **Section 3** | **Cloud Spanner Setup**: Idempotently provisions Cloud Spanner instance `{SPANNER_INSTANCE}` (`ENTERPRISE`, `100` PUs in `regional-{REGION}`) (`3.1`) and database `{SPANNER_DATABASE}` (`3.2`). |
| **Section 4** | **Create the Schema with `schema.sql`**: Displays `schema.sql` (`4.1`) and applies the DDL to `{SPANNER_INSTANCE}/{SPANNER_DATABASE}` (`4.2`), ensuring `Account.embedding ARRAY<FLOAT64>(vector_length=>128)` and `FraudGraph` are ready. |
| **Section 5** | **Load the Data from BigQuery with Reverse ETL (`EXPORT DATA`)**: Creates BigQuery `ENTERPRISE` reservation `{BQ_RESERVATION}` (`0` baseline, `50` autoscale max slots) and assigns `PIPELINE` and `QUERY` job types (`5.1`), exports parent Node tables first (`5.2`), and exports interleaved child Edge tables second (`5.3`) via `EXPORT DATA OPTIONS(format='CLOUD_SPANNER', spanner_options='{"table": "..."}')`. |
| **Section 6** | **Perform BigQuery Access to Cloud Spanner with `EXTERNAL_DATASET`**: Creates Zero-ETL external schema `{PROJECT_ID}.{BQ_SPANNER_EXTERNAL_DATASET}` (`6.1`), verifies all 6 live Spanner tables (`6.2`), and executes a 4-table federated country risk join via `%%bqsql` (`6.3`). |
| **Section 7** | **Explore All 6 Fraud Patterns & Student Challenges (`%%spanner_graph`)**: Runs visual `%%spanner_graph` queries (`RETURN SAFE_TO_JSON(query_path) AS path_json, ...`) for all 6 AML topologies, each paired with an interactive student challenge (solutions in `assets/CHALLENGE_SOLUTIONS.md`). |
| **Section 8** | **GNN — Train the Model & Create Embeddings (`dgf_gnn_pipeline.py`)**: Executes the 2-Hop Heterogeneous GraphSAGE pipeline (`dgf_gnn_pipeline.py`), writing 128-dim L2-normalized embeddings to `Account.embedding` (`Recall = 1.0000`, `Precision = 0.6593`, `F1 = 0.7947`). |
| **Section 9** | **Show GNN Examples — `COSINE_DISTANCE` & 2D PCA Visualization**: Finds the top-10 structural nearest neighbors of `SHELL_CORP` aggregator `20006` using `COSINE_DISTANCE` in `%%spanner_graph` (matching counterpart hub `20012` at `0.9573` similarity) and projects the embeddings into 2D using `bigframes.ml.decomposition.PCA(n_components=2)`. |
| **Section 10** | **Agentic Fraud Investigation via MCP (`mcp_fraud_server.py`)**: Calls `SpannerGraphClient` and `handle_tool_call(mcp_client, "investigate_account", {"account_id": 20006})` to generate a complete JSON compliance dossier. |

---

## 6. Verified Fraud Patterns & Student Challenges Summary

| # | Pattern Code | Fraudulent Accounts | Discovery Query Highlights (`%%spanner_graph`) | Student Challenge (`assets/CHALLENGE_SOLUTIONS.md`) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `MONEY_MULE_RING` | `12` (`20001..20012`) | Aggregators `20012` (`51,816.31 EUR` inflow -> `19,211.82 EUR` to `21002`) and `20006` (`45,849.24 EUR` inflow -> `16,945.24 EUR` to `21001`). | Trace 2 hops upstream to identify the original victim accounts (`20550` lost `4,365.22 EUR` -> mule `20007` -> hub `20012`). |
| **2** | `SHARED_FRAUD_DEVICE` | `10` (`20013..20022`) | Rooted `EMULATOR` devices `30001` and `30002` shared across `7` accounts each (`is_vpn_or_proxy = TRUE`). | Follow flagged drain transfers from `EMULATOR` accounts (`20013..20022`) to reveal they feed Pattern 1 hubs `20006` and `20012`. |
| **3** | `STRUCTURING_SMURFING` | `10` (`20023..20032`) | Collectors `20027` (`76,158.71 EUR` from smurfs `20023..20026` -> `75,397.12 EUR` wire) and `20032` (`75,473.09 EUR` from smurfs `20028..20031` -> `74,718.36 EUR` wire). | Traverse `(owner:Person)-[:Owns]->(smurf:Account)` to unmask the 8 KYC owners (`10023..10026`, `10028..10031`) behind the smurf accounts. |
| **4** | `RAPID_CYCLING` | `10` (`20033..20042`) | Pass-through accounts forwarding `16,182–27,422 EUR` within `5–11` minutes (`96.81%–98.06%` value pass-through). | Filter for `<= 10` minute dwell time (`7` accounts) and rank by skimmed fee (`20042` skimmed `738.55 EUR` in `5` minutes). |
| **5** | `IDENTITY_REUSE_MULE` | `8` (`20043..20050`) | Alpha (`10043..10046`) and Beta (`10047..10050`) rings covertly sharing `phone`, `email`, and `address` across "disconnected" `Person` records. | Extend path across `-[u:UsesDevice]->(d:Device {is_vpn_or_proxy: TRUE})` to link Alpha to VPN `30003` and Beta to VPN `30004` (`10.88.77.10..13`). |
| **6** | `CIRCULAR_LOOP` | `10` (`20051..20060`) | Quantified path `-[t:Transfer WHERE t.amount >= 10000.0]->{3, 4}` surfaces 4-hop ring `20051..20054` and 3-hop rings `20055..20057`, `20058..20060`. | Add chronological origin filter `t[0].create_time < t[ARRAY_LENGTH(t) - 1].create_time` to isolate true origins (`20058`, `20051`, `20055`) and fee attrition. |
