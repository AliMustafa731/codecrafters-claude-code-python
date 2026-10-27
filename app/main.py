import argparse
import os
import sys
import json
import subprocess
import re

from openai import OpenAI
import frontmatter

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL = os.getenv("AMK_LOCAL_MODEL") if os.getenv("AMK_LOCAL_MODEL") else "anthropic/claude-haiku-4.5"

def parse_and_inject_skills(prompt, available_skills):
    def replace_skill(match):
        skill_name = match.group(1)[1:]
        arguments_text = (match.group(2) or "").strip()
        arguments = arguments_text.split()

        # Leave unknown slash commands unchanged.
        if skill_name not in available_skills:
            return match.group(0)

        skill_body = available_skills[skill_name]["body"]

        def replace_placeholder(placeholder_match):
            placeholder = placeholder_match.group(1)

            if placeholder == "ARGUMENTS":
                return arguments_text

            if placeholder.startswith("ARGUMENTS["):
                index = int(placeholder[len("ARGUMENTS["):-1])
            else:
                index = int(placeholder)

            return arguments[index] if index < len(arguments) else ""

        return re.sub(
            r"\$(ARGUMENTS(?:\[\d+\])?|\d+)",
            replace_placeholder,
            skill_body,
        )

    # Skill arguments are everything after the skill name on that line.
    return re.sub(r"(?<!\S)(/\w+)(?:[ \t]+([^\n]*))?", replace_skill, prompt)

def load_skills(skills_path):
    skills = {}

    try:
        entries = os.scandir(skills_path)

        for entry in entries:
            if entry.is_dir():
                # initialize empty skill body
                skill_name = entry.name
                skill_file_path = os.path.join(skills_path, skill_name, "SKILL.md")
                post = frontmatter.load(skill_file_path)
                skills[skill_name] = {
                    "name": skill_name,
                    "description": post["description"],
                    "body": post.content
                }
    except FileNotFoundError:
        print(f"Error: The directory '{skills_path}' could not be found.", file=sys.stderr)

    return skills

def execute_tool_call(tool):
    content = ""

    if tool.type == "function":
        arguments  = json.loads(tool.function.arguments)

        if tool.function.name == "Read":
            with open(arguments["file_path"], "r", encoding="utf-8") as file:
                content = file.read()
                print(f"LOG: Successfully read the file [{arguments["file_path"]}]", file=sys.stderr)
        
        elif tool.function.name == "Write":
            with open(arguments["file_path"], "w", encoding="utf-8") as file:
                file.write(arguments["content"])
                content = f"Successfully written content to file [{arguments["file_path"]}"
                print(f"LOG: Successfully written content to file [{arguments["file_path"]}]", file=sys.stderr)
        
        elif tool.function.name == "Bash":
            print(f"[Executing System Command]: {arguments["command"]}")
            try:
                result = subprocess.run(
                    arguments["command"],
                    shell=True,
                    text=True,
                    capture_output=True
                )
                if result.returncode == 0:
                    content = result.stdout if result.stdout else "Command executed successfully (no output)."
                else:
                    content = f"Error (Exit Code {result.returncode}): {result.stderr}"
            except Exception as e:
                return f"Execution failed: {str(e)}"
            
            print(f"LOG: Executed the shell command [{arguments["command"]}]", file=sys.stderr)

    return content

def main():
    # parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # setup openai client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # setup system and user prompts
    skills = load_skills(os.path.join(".", ".claude", "skills"))

    system_prompt = "You have access to the following skills:\n\n"

    for skill in skills.values():
        system_prompt += f"- {skill["name"]}: {skill["description"]}\n"

    user_prompt = parse_and_inject_skills(args.p, skills)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    }
                }
            }
        }
    ]

    while True:
        chat = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        print("LOG: Received message from the model", file=sys.stderr)

        message = chat.choices[0].message;
        messages.append(message.model_dump())

        # execute tools
        if message.tool_calls:
            for tool in message.tool_calls:
                print(f"LOG: Received tool call [{tool.function.name}]", file=sys.stderr)
                result = execute_tool_call(tool)
                messages.append({"role": "tool", "content": result, "tool_call_id": tool.id})
        else:
            print("LOG: Final result printed", file=sys.stderr)
            print(message.content)
            break

if __name__ == "__main__":
    main()
