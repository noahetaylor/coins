from label_studio_sdk import Client

LS_URL = "http://localhost:8080"
API_KEY = "YOUR_API_KEY"
PROJECT_ID = 1
SOURCE_TASK_ID = 101  # the one you manually labeled

ls = Client(url=LS_URL, api_key=API_KEY)
project = ls.get_project(PROJECT_ID)

# Get the source annotation
source_task = project.get_task(SOURCE_TASK_ID)
source_annotation = source_task['annotations'][0]['result']

# Get all tasks in the project
all_tasks = project.get_tasks()

for task in all_tasks:
    if task['id'] == SOURCE_TASK_ID:
        continue

    project.create_annotation(
        task_id=task['id'],
        result=source_annotation,
        completed_by=1
    )

print("Bulk annotation applied.")
