############################################################
### Script to clean the venv of all non-default packages ###
############################################################

import subprocess


def get_installed_packages():
    """Fetches a list of all installed pip packages."""
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
    packages = result.stdout.splitlines()
    return packages


def filter_default_packages(packages):
    """Filters out default libraries that should not be uninstalled."""
    default_packages = {'setuptools', 'wheel', 'pip'}
    filtered_packages = [pkg for pkg in packages if not any(default in pkg for default in default_packages)]
    return filtered_packages


def write_to_file(filtered_packages):
    """Writes the filtered packages to uninstall_pip.txt."""
    with open('uninstall_pip.txt', 'w') as file:
        for package in filtered_packages:
            file.write(f"{package.split('==')[0]}\n")
    print("uninstall_pip.txt has been created.")


def uninstall_packages():
    """Reads uninstall_pip.txt and uninstalls the listed packages."""
    with open('uninstall_pip.txt', 'r') as file:
        packages = file.read().splitlines()
        if packages:
            for package in packages:
                subprocess.run(['pip', 'uninstall', '-y', package])
            print("All packages from uninstall_pip.txt have been uninstalled.")
        else:
            print("No packages to uninstall.")


if __name__ == '__main__':
    packages = get_installed_packages()
    filtered_packages = filter_default_packages(packages)
    write_to_file(filtered_packages)

    # Uncomment the line below to uninstall packages automatically
    uninstall_packages()
