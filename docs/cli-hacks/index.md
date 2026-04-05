# CLI Hacks

## Introduction to Command Line Productivity

The command line interface (CLI) is a powerful tool for developers, system administrators, and power users alike. While many modern applications provide graphical user interfaces (GUIs), mastering the CLI can significantly enhance your productivity and allow for more efficient workflows. In this post, we'll explore various CLI hacks that can streamline your tasks, automate repetitive processes, and enable advanced functionalities. 

Whether you are using Bash on Linux, Terminal on macOS, or PowerShell on Windows, the techniques discussed here can be applied across platforms. We will cover practical examples, specific tools, and use cases to help you harness the full potential of your command line.

## 1. Mastering Command History

### Use Case: Efficient Command Retrieval

When working in the terminal, you often find yourself repeating commands. Mastering command history can save you time and keystrokes.

#### **Command History Navigation**

- Use the `history` command to list all previous commands.
  ```bash
  history
  ```

- Use the `!` operator to execute a command from history quickly. For example, `!42` will run the 42nd command in your history.

- Navigate through your command history using the **up** and **down** arrow keys.

### **Enhancement: Fuzzy Search with `fzf`**

For an enhanced command history experience, consider using `fzf`, a command-line fuzzy finder.

#### **Installation:**
- On macOS, use Homebrew:
  ```bash
  brew install fzf
  ```

- On Ubuntu:
  ```bash
  sudo apt-get install fzf
  ```

#### **Usage:**
Once installed, you can invoke it by running:
```bash
history | fzf
```
This command will open an interactive interface to search through your command history. You can type a few letters to filter results, making it much faster to find and execute previous commands.

## 2. Command Chaining

### Use Case: Efficient Multi-Command Execution

Command chaining allows you to run multiple commands in a single line, reducing the need for repetitive typing.

#### **Basic Chaining: `&&` and `;`**

- **Using `&&`:** This operator runs the second command only if the first command succeeds.
  ```bash
  mkdir new_folder && cd new_folder
  ```

- **Using `;`:** This operator runs both commands regardless of the success of the first.
  ```bash
  echo "Starting process" ; sleep 5 ; echo "Process completed"
  ```

### **Advanced Chaining: Using `xargs`**

For commands that require input from the output of another command, `xargs` can be particularly powerful. 

#### **Example: Deleting Files**

Suppose you want to delete all `.tmp` files in a directory:
```bash
find . -name "*.tmp" | xargs rm
```
In this example, `find` locates all `.tmp` files and pipes the results to `xargs`, which executes `rm` on each file found.

### **Practical Application: Automating Backups**

You can create a backup script that combines several commands, checks for errors, and sends a notification.

```bash
#!/bin/bash
tar -czf backup.tar.gz /path/to/important/files && echo "Backup Successful" || echo "Backup Failed"
```
In this script, a compressed tarball is created, and based on the success of the command, a message is displayed.

## 3. Custom Aliases for Repetitive Tasks

### Use Case: Speeding Up Common Commands

Creating aliases for frequently used commands can drastically reduce typing time.

#### **Creating Aliases**

You can define your aliases in the shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`).

```bash
# Example aliases
alias ll='ls -la'
alias gs='git status'
alias gp='git pull'
```

#### **Reloading Configuration**

After adding your aliases, reload the configuration:
```bash
source ~/.bashrc  # For Bash
source ~/.zshrc   # For Zsh
```

### **Practical Application: Git Workflow Shortcuts**

If you frequently use Git, create aliases for common Git commands to speed up your workflow:

```bash
alias ga='git add'
alias gc='git commit -m'
alias gco='git checkout'
```
Now, instead of typing `git add .`, you can simply use `ga .`.

## 4. Process Management with `tmux`

### Use Case: Managing Multiple Sessions

`tmux` is a terminal multiplexer that allows you to run multiple terminal sessions within a single window. 

#### **Installation:**

- On macOS:
  ```bash
  brew install tmux
  ```

- On Ubuntu:
  ```bash
  sudo apt-get install tmux
  ```

#### **Basic Usage:**

- Start `tmux`:
  ```bash
  tmux
  ```

- Create a new window:
  ```bash
  Ctrl + b, c
  ```

- Switch between windows:
  ```bash
  Ctrl + b, n  # Next window
  Ctrl + b, p  # Previous window
  ```

- Detach from a session:
  ```bash
  Ctrl + b, d
  ```

You can reattach to a session using:
```bash
tmux attach-session -t [session_name]
```

### **Real-World Scenario: Running Long Processes**

If you're running a long-running process (like a build or a server), you can start it in `tmux` and detach from it. This allows you to close your terminal without stopping the process.

```bash
tmux new -s mysession
# Start a long-running task, like a server
python manage.py runserver
```
Detach from the session and close your terminal. You can later reattach:
```bash
tmux attach-session -t mysession
```

## 5. File Management with `find`, `grep`, and `sed`

### Use Case: Advanced File Searching and Manipulation

When dealing with large directories, finding files can be cumbersome. The `find`, `grep`, and `sed` commands can help streamline these tasks.

#### **Using `find` to Locate Files**

To find all `.log` files modified in the last 7 days:
```bash
find /path/to/search -name "*.log" -mtime -7
```

#### **Combining `find` with `grep`**

If you want to find lines containing "ERROR" in those log files:
```bash
find /path/to/search -name "*.log" -mtime -7 -exec grep -H "ERROR" {} \;
```
Here, `-exec` allows you to execute `grep` on each file found.

#### **Using `sed` for Replacement**

To replace "ERROR" with "WARNING" in a specific file:
```bash
sed -i 's/ERROR/WARNING/g' /path/to/file.log
```

### **Practical Application: Log File Management**

You can create a script to archive log files older than 30 days and compress them.

```bash
#!/bin/bash
find /var/log -name "*.log" -mtime +30 -exec tar -rvf old_logs.tar {} \; && echo "Archived old logs"
```

## 6. Automating Tasks with `cron`

### Use Case: Scheduled Automation

`cron` allows you to schedule commands to run at specific intervals. This is useful for automating system maintenance tasks.

#### **Editing the Crontab**

To edit the cron jobs for your user:
```bash
crontab -e
```

#### **Example of a Cron Job**

To run a backup script every day at 2 AM:
```bash
0 2 * * * /path/to/backup-script.sh
```

### **Cron Syntax Breakdown:**
- `0`: Minute (0-59)
- `2`: Hour (0-23)
- `*`: Day of month (1-31)
- `*`: Month (1-12)
- `*`: Day of week (0-7, where both 0 and 7 represent Sunday)

### **Monitoring Cron Jobs**

To check the status of your cron jobs, you can view the system logs. On most systems, cron logs are located in `/var/log/syslog`.

## 7. Networking with `curl` and `wget`

### Use Case: Downloading and Testing APIs

`curl` and `wget` are powerful tools for downloading files and testing web APIs.

#### **Using `curl`**

To fetch the contents of a webpage:
```bash
curl https://www.example.com
```

To test a REST API endpoint:
```bash
curl -X GET "https://api.example.com/data" -H "Authorization: Bearer YOUR_TOKEN"
```

#### **Using `wget`**

To download a file:
```bash
wget https://example.com/file.zip
```

To download an entire website for offline viewing:
```bash
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent https://example.com
```

### **Practical Application: API Monitoring Script**

You can create a simple script to check the availability of an API endpoint and log the response time.

```bash
#!/bin/bash
for i in {1..5}; do
    START=$(date +%s%3N)
    curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health
    END=$(date +%s%3N)
    DIFF=$((END - START))
    echo "Response Time: ${DIFF} ms"
    sleep 10
done
```

## Conclusion

Mastering the command line can drastically improve your productivity and efficiency. The hacks discussed in this post—from managing command history to creating automated tasks—are just the tip of the iceberg. Here’s a quick recap of actionable steps you can take:

1. **Enhance Command History:** Install `fzf` to make command retrieval faster.
2. **Optimize Command Chaining:** Use `&&`, `;`, and `xargs` for efficient command execution.
3. **Create Custom Aliases:** Reduce typing by creating aliases for frequently used commands.
4. **Utilize `tmux`:** Manage multiple terminal sessions effectively with `tmux`.
5. **Leverage `find`, `grep`, and `sed`:** Master file searching and manipulation.
6. **Automate with `cron`:** Schedule tasks for maintenance and backups.
7. **Download and Test with `curl` and `wget`:** Efficiently fetch resources and test APIs.

### Next Steps

- **Practice Regularly:** The more you use the CLI, the more comfortable you will become.
- **Explore Advanced Tools:** Investigate additional tools like `awk`, `jq`, and `httpie` to expand your capabilities.
- **Create Your Own Scripts:** Start automating your repetitive tasks using the techniques outlined above.

By implementing these CLI hacks, you’ll find that you can work smarter, not harder, and unlock the true power of your command line.