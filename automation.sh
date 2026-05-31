#!/bin/bash

AGENT_FILES=(
    "my_agent_3.py"
    "my_agent_copy_2.py"
    "my_agent_copy_3.py"
    "my_agent_copy.py"
    "my_agent_0.py"
    "my_agent_1.py"
    "my_agent_2_copy.py"
    "my_agent_2.py"
    "my_agent.py"
)

# Clear out.txt before starting
> out.txt

for agent_file in "${AGENT_FILES[@]}"; do
    # Extract the module name (remove .py and replace spaces/hyphens as needed for Python import)
    # This assumes your actual Python module names match the file names after basic sanitization.
    # If your internal module names differ, you'll need a more sophisticated mapping.
    module_name=$(basename "$agent_file" .py)
    module_name=${module_name// /_} # Replace spaces with underscores
    module_name=${module_name//-/_} # Replace hyphens with underscores

    echo "--- Running tests for: $agent_file (Module: $module_name) ---" | tee -a out.txt
    python testing.py "$module_name" 2>&1 | tee -a out.txt
    echo "" | tee -a out.txt # Add a blank line for separation
done

echo "All tests completed. Output redirected to out.txt"