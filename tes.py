import subprocess

def check_command(command):
    try:
        # Run the command and check if it is installed
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        return str(e)

# Check for Git
git_version = check_command(['git', '--version'])
if git_version:
    print(f"Git is installed: {git_version}")
else:
    print("Git is not installed.")

# Check for Git LFS
git_lfs_version = check_command(['git', 'lfs', 'version'])
if git_lfs_version:
    print(f"Git LFS is installed: {git_lfs_version}")
else:
    print("Git LFS is not installed.")
