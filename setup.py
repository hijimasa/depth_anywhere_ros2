import sys
import urllib.request
import zipfile
import glob
import os
from setuptools import find_packages, setup
import glob
import os
import shutil
from zipfile import ZipFile
import requests

package_name = 'depth_anywhere_ros2'

# Check if UniFuse directory exists in ckpt folder
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1yE555x5tvC3zJx_KxyuMKi4ok-joKpdg&export=download&authuser=0&confirm=t&uuid=9cd70cd3-82e1-4921-84cd-82add4216766&at=ALoNOglf-ccUjuZBaqROJcffZPJT%3A1747060462078"
ckpt_dir = os.path.join('ckpt')
unifuse_dir = os.path.join('ckpt', 'UniFuse')
if not os.path.exists(unifuse_dir):
    sys.stderr.write("UniFuse directory not found. Downloading checkpoint...\n")
        
    zip_path = os.path.join(ckpt_dir, 'checkpoint.zip')
    sys.stderr.write(f'Downloading checkpoint from {CHECKPOINT_URL} to {zip_path}...\n')
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)
    except Exception as e:
        sys.stderr.write(f'Error downloading checkpoint: {e}\n', file=sys.stderr)
        sys.exit(1)
    sys.stderr.write('Download complete. Extracting files...\n')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(ckpt_dir)
    except zipfile.BadZipFile as e:
        sys.stderr.write(f'Error unpacking zip file: {e}\n', file=sys.stderr)
        sys.exit(1)
    # ZIP を削除
    os.remove(zip_path)
    sys.stderr.write('Extraction complete and zip file removed.\n')
else:
    sys.stderr.write('UniFuse checkpoint already exists, skip download.\n')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all files from ckpt directories
        (os.path.join('share', package_name, 'ckpt'),
            [f for f in glob.glob('ckpt/**/*', recursive=True) if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "infer = depth_anywhere_ros2.infer:main",
        ],
    },
)
