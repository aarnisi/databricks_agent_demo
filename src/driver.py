!pip install databricks-sdk==0.52.0 mlflow==2.22.0 databricks-agents==0.22.0 uv --quiet
dbutils.library.restartPython()


%md
### Set up Agent monitoring using MLflow


dbutils.library.restartPython()

from agent import agent
#from agent import agent
agent.predict({"messages": [{"role": "user", "content": "What is the weather in Stockholm? Then find out how many customers are in Genie and reply with both of those"}]})


agent = WeatherAgent()
for event in agent.predict_stream(
    {"messages": [{"role": "user", "content": "What is the weather in Stockholm? Then ask how many clients there are and create a weather forecast for all of them"}]}
):
    print(event, "-----------\n")



# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from mlflow.models.auth_policy import SystemAuthPolicy, UserAuthPolicy, AuthPolicy

from mlflow.models.resources import (
  DatabricksVectorSearchIndex,
  DatabricksServingEndpoint,
  DatabricksSQLWarehouse,
  DatabricksFunction,
  DatabricksGenieSpace,
  DatabricksTable,
  DatabricksUCConnection
)

resources = [DatabricksServingEndpoint(endpoint_name='demo-gpt4o-mini')]
#DatabricksGenieSpace(genie_space_id='01f0178100571292bbf1df599335de43')


# Specify resources here for system authentication
system_auth_policy = SystemAuthPolicy(resources=resources)

# Specify the minimal set of API scopes needed for on-behalf-of-user authentication
	# When deployed, the agent can access Databricks resources and APIs
	# on behalf of the end user, but only via REST APIs that are covered by the list of
	# scopes below

user_auth_policy = UserAuthPolicy(
    api_scopes=[
        "dashboards.genie"
    ]
)

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "How many data points there are in Genie?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="linkedin_demo",
        python_model="agent.py",
        code_paths=["./src"],
        input_example=input_example,
        pip_requirements=[
            "mlflow==2.22.0",
            "databricks-sdk==0.52.0",
            "openai==1.69.0",
            "pydantic",
        ],
        auth_policy=AuthPolicy(
            system_auth_policy=system_auth_policy,
            user_auth_policy=user_auth_policy
        )
    )


mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/linkedin_demo",
    input_data={"messages": [{"role": "user", "content": "Who is the most profitable customer?"}]},
    env_manager="uv",
)


mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "ai_model_catalog"
schema = "deepseek"
model_name = "delete_me_linkedin_demo"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)


from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})


mlflow.models.predict(
    model_uri=f"models:/ai_model_catalog.deepseek.delete_me3_genie_test/2",
    input_data={"messages": [{"role": "user", "content": "Who is the most profitable customer?"}]},
    env_manager="uv",
)


from databricks.sdk import WorkspaceClient

messages = [{"role": "user", "content": "Who is the least profitable customer"}]
endpoint = "agents_ai_model_catalog-deepseek-delete_me3_genie_test"

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()
response = client.chat.completions.create(model=endpoint, messages=messages)
response
