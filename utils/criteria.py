import json

def get_criteria(task_type: int) -> str:
    """
    Retrieve band descriptors for the specified task type (1 or 2).

    Args:
        task_type (int): The task type (1 for Task 1, 2 for Task 2).

    Returns:
        str: The criteria string for the specified task.
    """
    if task_type not in [1, 2]:
        raise ValueError("Invalid task_type. Must be 1 or 2.")

    # Load the band descriptors from the JSON file
    with open('resource/band_assignments.json', 'r') as file:
        evaluation_criteria = json.load(file)

    # Select the appropriate key based on task type
    task_key = f"Writing Task {task_type} Band Descriptors"
    if task_key not in evaluation_criteria:
        raise KeyError(f"{task_key} not found in band_assignments.json")

    # Extract the band descriptors
    task_descriptors = evaluation_criteria[task_key]

    # Build the criteria string
    criteria_string = ""
    for band, descriptors in task_descriptors.items():
        criteria_string += f"Band {band}:\n"
        for category, description in descriptors.items():
            criteria_string += f"  {category}: {description}\n"

    return criteria_string
