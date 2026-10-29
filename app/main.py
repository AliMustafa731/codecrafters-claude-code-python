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

def inject_args_into_skill_body(arguments, skill):

    def replace_placeholder(placeholder_match):
        placeholder = placeholder_match.group(1)

        if placeholder == "ARGUMENTS":
            return " ".join(arguments)

        if placeholder.startswith("ARGUMENTS["):
            index = int(placeholder[len("ARGUMENTS["):-1])
        else:
            index = int(placeholder)

        return arguments[index] if index < len(arguments) else ""

    result = f"Any paths below are relative to (.claude/skills/{skill["name"]}).\n\n"

    if arguments:
        result += re.sub(r"\$(ARGUMENTS(?:\[\d+\])?|\d+)", replace_placeholder, skill["body"])
    else:
        result += skill["body"]

    return result

def parse_and_inject_skills(prompt, available_skills):
    arguments = prompt.split()
    skills = []
    results = []

    while arguments:
        if arguments[0].startswith("/"):
            if arguments[0][1:] in available_skills:
                skills.append(arguments.pop(0)[1:])
        else:
            break

    # now, the list "words" contains the arguments passed to the skills
    # now, we expand the skills bodies
    if skills:
        for skill_name in skills:
            skill_body = inject_args_into_skill_body(arguments, available_skills[skill_name])
            results.append(skill_body)
        
        return " ".join(results)
    else:
        return prompt

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

def execute_tool_call(tool, available_skills):
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
            print(f"[Executing System Command]: {arguments["command"]}", file=sys.stderr)
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
        
        elif tool.function.name == "Skill":
            skill_name = arguments.get("name")
            skill_args = arguments.get("args")

            return inject_args_into_skill_body(skill_args.split() if skill_args else None, available_skills[skill_name])

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

    system_prompt += "If a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns.\n"

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
        },
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load a skill's instructions into the conversation",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": { "type": "string", "description": "The name of the skill to use" },                                                            
                        "args": { "type": "string", "description": "Optional arguments for the skill" }                                                            
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

        print("[Assistant]", message.content, file=sys.stderr)

        # execute tools
        if message.tool_calls:
            for tool in message.tool_calls:
                print(f"LOG: Received tool call [{tool.function.name}]", file=sys.stderr)
                result = execute_tool_call(tool, skills)
                messages.append({"role": "tool", "content": result, "tool_call_id": tool.id})
        else:
            print("LOG: Final result printed", file=sys.stderr)
            print(message.content)
            break

if __name__ == "__main__":
    main()
