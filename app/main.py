import argparse
import os
import sys
import json
import subprocess
import re

from openai import OpenAI
import frontmatter

# Global parameters
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL = os.getenv("AMK_LOCAL_MODEL") if os.getenv("AMK_LOCAL_MODEL") else "anthropic/claude-haiku-4.5"

TOOLS = [
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

# Functions
def log(string):
    print(string, file=sys.stderr)

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
                skills[skill_name] = {**post, "body": post.content}
    except FileNotFoundError:
        log(f"Error: The directory '{skills_path}' could not be found.")

    return skills

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
available_skills = load_skills(os.path.join(".", ".claude", "skills"))

def parse_user_prompt(prompt):
    arguments = prompt.split()
    skills = []
    results = []

    while arguments:
        if arguments[0].startswith("/"):
            if arguments[0][1:] in available_skills:
                skills.append(arguments.pop(0)[1:])
        else:
            break

    # Now, we expand the skills bodies
    if skills:
        for skill_name in skills:
            results.append(
                expand_skill_body(arguments, available_skills[skill_name])
            )
        
        return "".join(results)
    else:
        return prompt

def run_agent(messages):
    result = ""
    
    while True:
        chat = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        log("LOG: Received message from the model")

        message = chat.choices[0].message;
        messages.append(message.model_dump())

        log(f"[Assistant] {message.content}")

        # execute tools
        if message.tool_calls:
            for tool in message.tool_calls:
                log(f"LOG: Received tool call [{tool.function.name}]")
                result = execute_tool_call(tool)
                messages.append({"role": "tool", "content": result, "tool_call_id": tool.id})
        else:
            log("LOG: Final result printed")
            result = message.content
            break

    return result

# Functions
def expand_skill_body(arguments, skill):

    def replace_placeholder(placeholder_match):
        placeholder = placeholder_match.group(1)

        if placeholder == "ARGUMENTS":
            return " ".join(arguments)

        if placeholder.startswith("ARGUMENTS["):
            index = int(placeholder[len("ARGUMENTS["):-1])
        else:
            index = int(placeholder)

        return arguments[index] if index < len(arguments) else ""

    skill_body = skill["body"]
    skill_name = skill["name"]

    if arguments:
        skill_body += re.sub(r"\$(ARGUMENTS(?:\[\d+\])?|\d+)", replace_placeholder, skill["body"])

    skill_body = f"Skill: {skill_name} (located at .claude/skills/{skill_name}).\nPaths in the instructions below are relative to that folder\nHere are the instructions for this skill:\n{skill_body}\n"

    # Run in a subagent if specified by the skill
    if skill.get("context") == "fork":
        result = run_agent([{"role": "user", "content": skill_body}])
        return f"Skill {skill_name} ran in a separate context and returned: {result}\n"

    return skill_body

def execute_tool_call(tool):
    content = ""

    if tool.type == "function":
        arguments  = json.loads(tool.function.arguments)

        if tool.function.name == "Read":
            with open(arguments["file_path"], "r", encoding="utf-8") as file:
                content = file.read()
                log(f"LOG: Successfully read the file [{arguments["file_path"]}]")
        
        elif tool.function.name == "Write":
            with open(arguments["file_path"], "w", encoding="utf-8") as file:
                file.write(arguments["content"])
                content = f"Successfully written content to file [{arguments["file_path"]}"
                log(f"LOG: Successfully written content to file [{arguments["file_path"]}]")
        
        elif tool.function.name == "Bash":
            log(f"[Executing System Command]: {arguments["command"]}")
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
            
            log(f"LOG: Executed the shell command [{arguments["command"]}]")
        
        elif tool.function.name == "Skill":
            skill_name = arguments.get("name")
            skill_args = arguments.get("args")

            content = expand_skill_body(skill_args.split() if skill_args else None, available_skills[skill_name])

    return content

def main():
    # Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # Setup system and user prompts
    system_prompt = "You have access to the following skills:\n\n"

    for skill in available_skills.values():
        system_prompt += f"- {skill["name"]}: {skill["description"]}\n"

    system_prompt += "If a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns.\n"

    user_prompt = parse_user_prompt(args.p)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Start the main agent
    result = run_agent(messages)
    print(result)

if __name__ == "__main__":
    main()
