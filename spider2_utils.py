import os
import pandas as pd
import re
import math
import copy

def load_csv_database(database_name, db_base_path=None, rows_limit=10, as_dict=False):
    """
    Load a CSV-dumped database into a dictionary where each key is a table name and the value is a pandas DataFrame.

    :param database_path: Path to the directory containing the CSV files representing the database.
    :return: A dictionary with table names as keys and pandas DataFrames as values.
    """
    DB_CSVS_BASE_PATH = db_base_path or os.getenv("DB_CSVS_BASE_PATH")
    database_path = os.path.join(DB_CSVS_BASE_PATH, "csv_dbs", database_name)

    table_dtype_df = pd.read_pickle(os.path.join(DB_CSVS_BASE_PATH, "dtype_mappings.pickle"))
    this_map = table_dtype_df[table_dtype_df["db_name"] == database_name]
    if len(this_map) == 1:
        dtype_map = this_map.iloc[0]["d_types"]
    else:
        dtype_map = {}
    
    tables = {}
    for file_name in os.listdir(database_path):
        if file_name.endswith(".csv"):
            table_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(database_path, file_name)
            dtypes = dtype_map.get(table_name, {})
            tables[table_name] = pd.read_csv(file_path)# dtype=dtypes
            if rows_limit >= 0:
                tables[table_name] = tables[table_name].iloc[:rows_limit]
            if as_dict:
                tables[table_name] = tables[table_name].to_dict(orient='records')
    return tables


def get_db_description(input_dict, input_dtypes, db_name=""):
    """
    Generates a description of the database structure, including tables, columns, and their data types.
    """
    description = f"The following is a list of tables in the {db_name} database, along with a list of the columns in the table, along with their types, where available, in parentheses.\n\n"
    
    for table_name, sample_rows in input_dict.items():
        description += f"Table: {table_name}\nColumns: "
        if table_name in input_dtypes:
            for column_name, dtype in input_dtypes[table_name].items():
                description += f"{column_name} ({dtype}), "
        else:
            description += "    No column information available.\n"
        description += "\n\n"
    return description

def make_header(db_name, work_dir=""):
    header = f"""
import pandas as pd

{db_name} = dict()
for table, table_data in load_csv_database('{db_name}', rows_limit=-1).items():
    {db_name}[table] = pd.DataFrame(table_data)
OUTPUT_DIR = f"{work_dir}/output.csv"
"""
    return header

def parse_generated_steps(llm_response):
    """
    Parses a string response from an LLM into a list of steps using regex.

    Args:
        llm_response (str): The LLM-generated response containing a numbered list of steps.

    Returns:
        list: A list of steps as strings.
    """
    # Use regex to match lines starting with a number followed by a period and a space
    step_pattern = re.compile(r'^\d+\.{0,1}\s+(.*)', flags=re.MULTILINE)
    steps = step_pattern.findall(llm_response)
    return steps

def get_table_dtype_map(database_name, table_name):
    table_dtype_df = pd.read_pickle("/kaggle/input/spider-dbs-csv/dtype_mappings.pickle")
    this_map = table_dtype_df[table_dtype_df["db_name"] == database_name]
    if len(this_map) == 1:
        dtype_map = this_map.iloc[0]["d_types"]
        return dtype_map.get(table_name, {})
    else:
        return {}


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=True):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    # print('condition_cols', condition_cols)
    
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    
    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    matches = 0
    matched = set()
    for k, gold in enumerate(t_gold_list):
        # if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
        #     score = 0
        # else:
        for j, pred in enumerate(t_pred_list):
            if k in matched:
                continue
            if vectors_match(gold, pred, ignore_order_=ignore_order):
                matches+=1
                matched.add(k)
                
    structure_score = len(set(gold_cols.columns).intersection(set(pred_cols.columns)))/len(set(gold_cols.columns))
    return structure_score, len(matched)/len(t_gold_list)


def compare_execution_context(exec_context, gold_value):
    # exec_context = copy.deepcopy(exec_context)
    # Convert Series to DataFrame for comparison
    for var_name, var_value in exec_context.items():
        if isinstance(var_value, pd.Series):
            exec_context[var_name] = var_value.to_frame()
    max_data_score = 0.0
    structure_score = 0.0
    for var_name, var_value in exec_context.items():
        try:
            # If the gold_value is a single row and single column, compare against primitives
            if isinstance(gold_value, pd.DataFrame) and gold_value.shape == (1, 1):
                gold_primitive = gold_value.iloc[0, 0]
                if isinstance(var_value, (int, float, str, bool)) and var_value == gold_primitive:
                    max_data_score = max(max_data_score, 1.0)
                    structure_score = 1.0
            # If both are DataFrames, use compare_pandas_table
            elif isinstance(var_value, pd.DataFrame) and isinstance(gold_value, pd.DataFrame):
                structure_score, data_score = compare_pandas_table(var_value, gold_value)
                max_data_score = max(max_data_score, data_score)
        except Exception as e:
            pass  # Ignore errors and continue
    return structure_score, max_data_score