import asyncio
import uuid
from gads.tools.sandbox import SandboxClient

async def run_diagnostics():
    project_id = uuid.UUID('aba9a430-23d2-4e54-84ef-21fe33117951')
    session_id = str(project_id)
    
    # Initialize the sandbox client
    client = SandboxClient(base_url="http://localhost:8000")
    
    # Diagnostics code to run in the sandbox
    code = """
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

# 1. Load data
df = pd.read_csv('creditcard.csv')
df = df.sample(50000, random_state=42).reset_index(drop=True)
target_col = 'Class'
problem_type = 'multiclass'
drop_cols = ['Time', 'Unnamed: 23']
df_clean = df.drop(columns=drop_cols, errors='ignore')

try:
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean[target_col])
except ValueError:
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)

# 2. Load the predictor
predictor = joblib.load('model.joblib')

# 3. Compute feature importance with subsample_size=1000 (default in the task)
print('--- Permutation Importance (subsample_size=1000, metric=accuracy) ---')
fi_sub = predictor.feature_importance(df_test, subsample_size=1000, num_shuffle_sets=1, silent=True)
print(fi_sub['importance'].head(10))

# 4. Check if there are any Class 1 instances in a random subsample of 1000 rows
sub_df = df_test.sample(n=1000, random_state=42)
print('\\nClass 1 count in 1000-row subsample:', (sub_df[target_col] == 1).sum())

# 5. Compute feature importance using the full df_test (size 10000)
print('\\n--- Permutation Importance (full test set 10000 rows, metric=accuracy) ---')
fi_full = predictor.feature_importance(df_test, num_shuffle_sets=1, silent=True)
print(fi_full['importance'].head(10))

# 6. Compute feature importance using roc_auc metric on full test set
print('\\n--- Permutation Importance (full test set, metric=roc_auc) ---')
try:
    # AutoGluon requires eval_metric to match problem type, but let's see if we can compute it on the test set.
    # We might need to specify the problem type or metric.
    # Let's inspect available metrics:
    print('Available metrics:', predictor.eval_metric)
except Exception as e:
    print('Failed to get metric:', e)
"""
    
    print("Executing diagnostic code in sandbox...")
    res = await client.execute(code, project_id=project_id, session_id=session_id)
    print("--- SANDBOX STDOUT ---")
    print(res.stdout)
    print("--- SANDBOX STDERR ---")
    print(res.stderr)
    if res.error:
        print("--- SANDBOX ERROR ---")
        print(res.error)
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
