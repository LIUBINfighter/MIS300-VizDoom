import os
from pathlib import Path

def get_files():
    root = Path(".")
    files_to_include = []

    # 1. Specific files
    specific_files = [
        "README.md",
        "docker-compose.yml",
        "main.py",
        "poetry.lock"
    ]
    for f in specific_files:
        path = root / f
        if path.exists():
            files_to_include.append(path)

    # 2. .py files in ./src recursively
    src_dir = root / "src"
    if src_dir.exists():
        files_to_include.extend(sorted(src_dir.rglob("*.py")))

    # 3. .md files in ./docs/dev recursively
    docs_dev_dir = root / "docs" / "dev"
    if docs_dev_dir.exists():
        files_to_include.extend(sorted(docs_dev_dir.rglob("*.md")))

    return files_to_include

def main():
    files = get_files()
    output_file = "codemix_output.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Write the list of relative paths first
        f.write("## 所有文件的相对路径\n\n")
        for file_path in files:
            f.write(f"- {file_path.as_posix()}\n")
        
        f.write("\n---\n\n")

        # Write content for each file
        for file_path in files:
            f.write(f"## {file_path.as_posix()}\n\n")
            
            # Determine language for markdown code block
            ext = file_path.suffix.lower()
            lang = ""
            if ext == ".py":
                lang = "python"
            elif ext == ".md":
                lang = "markdown"
            elif ext == ".yml" or ext == ".yaml":
                lang = "yaml"
            elif ext == ".lock":
                lang = "toml" # poetry.lock is toml-like
            
            f.write(f"```{lang}\n")
            try:
                with open(file_path, "r", encoding="utf-8") as content_f:
                    f.write(content_f.read())
            except Exception as e:
                f.write(f"Error reading file: {e}")
            f.write("\n```\n\n")
            f.write("————————————————————\n\n")

    print(f"Successfully merged {len(files)} files into {output_file}")

if __name__ == "__main__":
    main()
