{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "377c85a5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:49.034379Z",
     "iopub.status.busy": "2025-05-16T16:19:49.033882Z",
     "iopub.status.idle": "2025-05-16T16:19:49.038973Z",
     "shell.execute_reply": "2025-05-16T16:19:49.038132Z"
    },
    "papermill": {
     "duration": 0.01135,
     "end_time": "2025-05-16T16:19:49.040547",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.029197",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# from openai import AsyncOpenAI\n",
    "# from anthropic import AsyncAnthropic\n",
    "# import google.generativeai as genai\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "98bc07f3",
   "metadata": {
    "_cell_guid": "54c7fec2-96f3-4a68-a91a-ba65d88d357c",
    "_uuid": "0e29356d-8c4f-4936-a228-2b14b538c172",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:49.048136Z",
     "iopub.status.busy": "2025-05-16T16:19:49.047764Z",
     "iopub.status.idle": "2025-05-16T16:19:49.934410Z",
     "shell.execute_reply": "2025-05-16T16:19:49.933519Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.892456,
     "end_time": "2025-05-16T16:19:49.936401",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.043945",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import re\n",
    "import math\n",
    "import copy\n",
    "\n",
    "\n",
    "def load_csv_database(database_name, db_base_path, rows_limit=10, as_dict=False):\n",
    "    \"\"\"\n",
    "    Load a CSV-dumped database into a dictionary where each key is a table name and the value is a pandas DataFrame.\n",
    "\n",
    "    :param database_path: Path to the directory containing the CSV files representing the database.\n",
    "    :return: A dictionary with table names as keys and pandas DataFrames as values.\n",
    "    \"\"\"\n",
    "    DB_CSVS_BASE_PATH = db_base_path or os.getenv(\"DB_CSVS_BASE_PATH\")\n",
    "    database_path = os.path.join(DB_CSVS_BASE_PATH, \"csv_dbs\", database_name)\n",
    "\n",
    "    table_dtype_df = pd.read_pickle(os.path.join(DB_CSVS_BASE_PATH, \"dtype_mappings.pickle\"))\n",
    "    this_map = table_dtype_df[table_dtype_df[\"db_name\"] == database_name]\n",
    "    if len(this_map) == 1:\n",
    "        dtype_map = this_map.iloc[0][\"d_types\"]\n",
    "    else:\n",
    "        dtype_map = {}\n",
    "    \n",
    "    tables = {}\n",
    "    for file_name in os.listdir(database_path):\n",
    "        if file_name.endswith(\".csv\"):\n",
    "            table_name = os.path.splitext(file_name)[0]\n",
    "            file_path = os.path.join(database_path, file_name)\n",
    "            dtypes = dtype_map.get(table_name, {})\n",
    "            tables[table_name] = pd.read_csv(file_path)# dtype=dtypes\n",
    "            if rows_limit >= 0:\n",
    "                tables[table_name] = tables[table_name].iloc[:rows_limit]\n",
    "            if as_dict:\n",
    "                tables[table_name] = tables[table_name].to_dict(orient='records')\n",
    "    return tables\n",
    "\n",
    "\n",
    "def get_db_description(input_dict, input_dtypes, db_name=\"\"):\n",
    "    \"\"\"\n",
    "    Generates a description of the database structure, including tables, columns, and their data types.\n",
    "    \"\"\"\n",
    "    description = f\"The following is a list of tables in the {db_name} database, along with a list of the columns in the table, along with their types, where available, in parentheses.\\n\\n\"\n",
    "    \n",
    "    for table_name, sample_rows in input_dict.items():\n",
    "        description += f\"Table: {table_name}\\nColumns: \"\n",
    "        if table_name in input_dtypes:\n",
    "            for column_name, dtype in input_dtypes[table_name].items():\n",
    "                description += f\"{column_name} ({dtype}), \"\n",
    "        else:\n",
    "            description += \"    No column information available.\\n\"\n",
    "        description += \"\\n\\n\"\n",
    "    return description\n",
    "\n",
    "\n",
    "\n",
    "def make_header(db_name, work_dir=\"\"):\n",
    "    header = f\"\"\"\n",
    "import pandas as pd\n",
    "\n",
    "\n",
    "\n",
    "{db_name} = dict()\n",
    "for table, table_data in load_csv_database('{db_name}', rows_limit=-1).items():\n",
    "    {db_name}[table] = pd.DataFrame(table_data)\n",
    "OUTPUT_DIR = f\"{work_dir}/output.csv\"\n",
    "\"\"\"\n",
    "    return header\n",
    "\n",
    "\n",
    "\n",
    "def parse_generated_steps(llm_response):\n",
    "    \"\"\"\n",
    "    Parses a string response from an LLM into a list of steps using regex.\n",
    "\n",
    "    Args:\n",
    "        llm_response (str): The LLM-generated response containing a numbered list of steps.\n",
    "\n",
    "    Returns:\n",
    "        list: A list of steps as strings.\n",
    "    \"\"\"\n",
    "    # Use regex to match lines starting with a number followed by a period and a space\n",
    "    step_pattern = re.compile(r'^\\d+\\.{0,1}\\s+(.*)', flags=re.MULTILINE)\n",
    "    steps = step_pattern.findall(llm_response)\n",
    "    return steps\n",
    "\n",
    "\n",
    "\n",
    "# async def call_llm(provider, prompt=\"\", model=None, temperature=0.0, max_tokens=512, messages=None):\n",
    "#     try:\n",
    "#         messages = messages or [{\"role\": \"user\", \"content\": prompt}]\n",
    "#         if provider.lower() == \"openai\":\n",
    "#             client = AsyncOpenAI(api_key=os.getenv(\"OPENAI_API_KEY\"))\n",
    "#             response = await client.chat.completions.create(\n",
    "#                     model=model or \"gpt-4o-mini\",\n",
    "#                     store=False,\n",
    "#                     messages=messages,\n",
    "#                     temperature=temperature,\n",
    "#                     max_tokens=max_tokens\n",
    "#                 )\n",
    "#             return response.choices[0].message.content\n",
    "\n",
    "#         elif provider.lower() == \"anthropic\":\n",
    "#             anthropic_client = AsyncAnthropic(api_key=os.getenv(\"ANTHROPIC_API_KEY\"))\n",
    "#             response = await anthropic_client.messages.create(\n",
    "#                 model=model or \"claude-3-5-haiku-20241022\",\n",
    "#                 messages=messages,\n",
    "#                 temperature=temperature,\n",
    "#                 max_tokens=max_tokens,\n",
    "#             )\n",
    "#             return response.content[0].text\n",
    "\n",
    "#         elif provider.lower() == \"google\":\n",
    "#             gemini_model = genai.GenerativeModel(\n",
    "#                 model_name=model or \"gemini-2.0-flash-thinking-exp-01-21\",\n",
    "#                 generation_config={\n",
    "#                     \"temperature\": temperature,\n",
    "#                     \"top_p\": 0.95,\n",
    "#                     \"top_k\": 40,\n",
    "#                     \"max_output_tokens\": max_tokens,\n",
    "#                     \"response_mime_type\": \"text/plain\",\n",
    "#                 }\n",
    "#             )\n",
    "#             response = await gemini_model.generate_content_async(prompt)\n",
    "#             # time.sleep(6)\n",
    "#             return response.text\n",
    "\n",
    "#         else:\n",
    "#             return \"Unknown provider.\"\n",
    "\n",
    "#     except Exception as e:\n",
    "#         return f\"Error: {e}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9b6be2f3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:49.943693Z",
     "iopub.status.busy": "2025-05-16T16:19:49.943247Z",
     "iopub.status.idle": "2025-05-16T16:19:49.948243Z",
     "shell.execute_reply": "2025-05-16T16:19:49.947054Z"
    },
    "papermill": {
     "duration": 0.010449,
     "end_time": "2025-05-16T16:19:49.949964",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.939515",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def get_table_dtype_map(database_name, table_name):\n",
    "    table_dtype_df = pd.read_pickle(\"/kaggle/input/spider-dbs-csv/dtype_mappings.pickle\")\n",
    "    this_map = table_dtype_df[table_dtype_df[\"db_name\"] == database_name]\n",
    "    if len(this_map) == 1:\n",
    "        dtype_map = this_map.iloc[0][\"d_types\"]\n",
    "        return dtype_map.get(table_name, {})\n",
    "    else:\n",
    "        return {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "af0b2e88",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:49.956902Z",
     "iopub.status.busy": "2025-05-16T16:19:49.956601Z",
     "iopub.status.idle": "2025-05-16T16:19:49.966139Z",
     "shell.execute_reply": "2025-05-16T16:19:49.965155Z"
    },
    "papermill": {
     "duration": 0.01484,
     "end_time": "2025-05-16T16:19:49.967821",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.952981",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=True):\n",
    "    \"\"\"_summary_\n",
    "\n",
    "    Args:\n",
    "        pred (Dataframe): _description_\n",
    "        gold (Dataframe): _description_\n",
    "        condition_cols (list, optional): _description_. Defaults to [].\n",
    "        ignore_order (bool, optional): _description_. Defaults to False.\n",
    "\n",
    "    \"\"\"\n",
    "    # print('condition_cols', condition_cols)\n",
    "    \n",
    "    tolerance = 1e-2\n",
    "\n",
    "    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):\n",
    "        if ignore_order_:\n",
    "            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),\n",
    "                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))\n",
    "        if len(v1) != len(v2):\n",
    "            return False\n",
    "        for a, b in zip(v1, v2):\n",
    "            if pd.isna(a) and pd.isna(b):\n",
    "                continue\n",
    "            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):\n",
    "                if not math.isclose(float(a), float(b), abs_tol=tol):\n",
    "                    return False\n",
    "            elif a != b:\n",
    "                return False\n",
    "        return True\n",
    "    \n",
    "    if condition_cols != []:\n",
    "        gold_cols = gold.iloc[:, condition_cols]\n",
    "    else:\n",
    "        gold_cols = gold\n",
    "    pred_cols = pred\n",
    "\n",
    "    t_gold_list = gold_cols.transpose().values.tolist()\n",
    "    t_pred_list = pred_cols.transpose().values.tolist()\n",
    "    score = 1\n",
    "    matches = 0\n",
    "    matched = set()\n",
    "    for k, gold in enumerate(t_gold_list):\n",
    "        # if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):\n",
    "        #     score = 0\n",
    "        # else:\n",
    "        for j, pred in enumerate(t_pred_list):\n",
    "            if k in matched:\n",
    "                continue\n",
    "            if vectors_match(gold, pred, ignore_order_=ignore_order):\n",
    "                matches+=1\n",
    "                matched.add(k)\n",
    "                \n",
    "    structure_score = len(set(gold_cols.columns).intersection(set(pred_cols.columns)))/len(set(gold_cols.columns))\n",
    "    return structure_score, len(matched)/len(t_gold_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a5af1dbe",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:49.974871Z",
     "iopub.status.busy": "2025-05-16T16:19:49.974553Z",
     "iopub.status.idle": "2025-05-16T16:19:49.980817Z",
     "shell.execute_reply": "2025-05-16T16:19:49.980009Z"
    },
    "papermill": {
     "duration": 0.011571,
     "end_time": "2025-05-16T16:19:49.982460",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.970889",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def compare_execution_context(exec_context, gold_value):\n",
    "    # exec_context = copy.deepcopy(exec_context)\n",
    "    # Convert Series to DataFrame for comparison\n",
    "    for var_name, var_value in exec_context.items():\n",
    "        if isinstance(var_value, pd.Series):\n",
    "            exec_context[var_name] = var_value.to_frame()\n",
    "    max_data_score = 0.0\n",
    "    structure_score = 0.0\n",
    "    for var_name, var_value in exec_context.items():\n",
    "        try:\n",
    "            # If the gold_value is a single row and single column, compare against primitives\n",
    "            if isinstance(gold_value, pd.DataFrame) and gold_value.shape == (1, 1):\n",
    "                gold_primitive = gold_value.iloc[0, 0]\n",
    "                if isinstance(var_value, (int, float, str, bool)) and var_value == gold_primitive:\n",
    "                    max_data_score = max(max_data_score, 1.0)\n",
    "                    structure_score = 1.0\n",
    "            # If both are DataFrames, use compare_pandas_table\n",
    "            elif isinstance(var_value, pd.DataFrame) and isinstance(gold_value, pd.DataFrame):\n",
    "                structure_score, data_score = compare_pandas_table(var_value, gold_value)\n",
    "                max_data_score = max(max_data_score, data_score)\n",
    "        except Exception as e:\n",
    "            pass  # Ignore errors and continue\n",
    "    return structure_score, max_data_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "43f4498d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:49.989468Z",
     "iopub.status.busy": "2025-05-16T16:19:49.989161Z",
     "iopub.status.idle": "2025-05-16T16:19:49.992746Z",
     "shell.execute_reply": "2025-05-16T16:19:49.991959Z"
    },
    "papermill": {
     "duration": 0.008811,
     "end_time": "2025-05-16T16:19:49.994302",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.985491",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# q =  load_csv_database(\"E_commerce\", \"/kaggle/input/spider-dbs-csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c29d8fde",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:50.001016Z",
     "iopub.status.busy": "2025-05-16T16:19:50.000657Z",
     "iopub.status.idle": "2025-05-16T16:19:50.004307Z",
     "shell.execute_reply": "2025-05-16T16:19:50.003360Z"
    },
    "papermill": {
     "duration": 0.00856,
     "end_time": "2025-05-16T16:19:50.005821",
     "exception": false,
     "start_time": "2025-05-16T16:19:49.997261",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# q.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "745f6f47",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-16T16:19:50.013861Z",
     "iopub.status.busy": "2025-05-16T16:19:50.013526Z",
     "iopub.status.idle": "2025-05-16T16:19:50.017262Z",
     "shell.execute_reply": "2025-05-16T16:19:50.016234Z"
    },
    "papermill": {
     "duration": 0.008874,
     "end_time": "2025-05-16T16:19:50.018885",
     "exception": false,
     "start_time": "2025-05-16T16:19:50.010011",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# compare_pandas_table(q['order_payments'], q['order_reviews'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a74dc04",
   "metadata": {
    "papermill": {
     "duration": 0.002558,
     "end_time": "2025-05-16T16:19:50.024532",
     "exception": false,
     "start_time": "2025-05-16T16:19:50.021974",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 7078100,
     "sourceId": 11393376,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30918,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 4.247801,
   "end_time": "2025-05-16T16:19:50.548399",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-05-16T16:19:46.300598",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
