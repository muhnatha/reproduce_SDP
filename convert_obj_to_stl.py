#!/usr/bin/env python3
"""
Script to convert all OBJ mesh files to STL format in the shapenet_core directory
and update XML file references accordingly.
"""

import os
import glob
from pathlib import Path
import re
import shutil
import argparse

try:
    import trimesh
except ImportError:
    print("Error: trimesh package not found. Install with: pip install trimesh")
    exit(1)


def convert_obj_to_stl(obj_path, stl_path):
    """Convert a single OBJ file to STL format."""
    try:
        mesh = trimesh.load(obj_path)
        mesh.export(stl_path)
        return True, None
    except Exception as e:
        return False, str(e)


def find_obj_files(root_dir):
    """Find all OBJ files in the directory tree."""
    obj_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.obj'):
                obj_files.append(os.path.join(root, f))
    return obj_files


def find_xml_files(root_dir):
    """Find all XML files in the directory tree."""
    xml_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.xml'):
                xml_files.append(os.path.join(root, f))
    return xml_files


def update_xml_file_references(xml_path, backup_dir):
    """Update XML file to replace .obj references with .stl."""
    with open(xml_path, 'r') as f:
        content = f.read()

    original_content = content

    # Replace .obj with .stl in mesh file references
    content = content.replace('.obj"', '.stl"')

    if content == original_content:
        return False, "No .obj references found"

    # Create backup
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(xml_path) + '.backup')
    with open(backup_path, 'w') as f:
        f.write(original_content)

    # Write updated content
    with open(xml_path, 'w') as f:
        f.write(content)

    return True, backup_path


def main():
    parser = argparse.ArgumentParser(description='Convert OBJ meshes to STL and update XML references')
    parser.add_argument('--root-dir', 
                       default='/home/cc/reproduce_SDP/mimicgen_environments/mimicgen/models/robosuite/assets/shapenet_core',
                       help='Root directory containing OBJ files and XML files')
    parser.add_argument('--backup-dir',
                       default='/home/cc/reproduce_SDP/mesh_conversion_backups',
                       help='Directory to store backup files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')

    args = parser.parse_args()

    root_dir = args.root_dir
    backup_dir = args.backup_dir

    if not os.path.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist")
        exit(1)

    print("=" * 70)
    print("OBJ to STL Conversion Script")
    print("=" * 70)
    print(f"Root directory: {root_dir}")
    print(f"Backup directory: {backup_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Find all OBJ files
    print("Step 1: Finding OBJ files...")
    obj_files = find_obj_files(root_dir)
    print(f"Found {len(obj_files)} OBJ files")

    # Find all XML files
    print("\nStep 2: Finding XML files...")
    xml_files = find_xml_files(root_dir)
    print(f"Found {len(xml_files)} XML files")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - No changes will be made")
        print("=" * 70)
        print("\nFirst 10 OBJ files to convert:")
        for f in obj_files[:10]:
            print(f"  {f}")
        print(f"\n... and {len(obj_files) - 10} more files")

        print("\nFirst 5 XML files to update:")
        for f in xml_files[:5]:
            print(f"  {f}")
        print(f"\n... and {len(xml_files) - 5} more files")
        return

    # Convert OBJ files to STL
    print("\nStep 3: Converting OBJ files to STL...")
    success_count = 0
    error_count = 0
    errors = []

    for obj_path in obj_files:
        stl_path = obj_path.replace('.obj', '.stl')

        # Skip if STL already exists and is newer than OBJ
        if os.path.exists(stl_path):
            if os.path.getmtime(stl_path) >= os.path.getmtime(obj_path):
                success_count += 1
                continue

        success, error = convert_obj_to_stl(obj_path, stl_path)
        if success:
            success_count += 1
            print(f"  ✓ {os.path.basename(obj_path)} -> {os.path.basename(stl_path)}")
        else:
            error_count += 1
            errors.append((obj_path, error))
            print(f"  ✗ {os.path.basename(obj_path)}: {error}")

    print(f"\nConversion complete: {success_count} succeeded, {error_count} failed")

    if errors:
        print("\nErrors:")
        for obj_path, error in errors[:5]:
            print(f"  {obj_path}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    # Update XML files
    print("\nStep 4: Updating XML file references...")
    xml_backup_dir = os.path.join(backup_dir, 'xml')
    xml_updated = 0
    xml_skipped = 0

    for xml_path in xml_files:
        updated, result = update_xml_file_references(xml_path, xml_backup_dir)
        if updated:
            xml_updated += 1
            print(f"  ✓ Updated {os.path.basename(xml_path)} (backup: {os.path.basename(result)})")
        else:
            xml_skipped += 1

    print(f"\nXML update complete: {xml_updated} updated, {xml_skipped} skipped (no .obj references)")

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print(f"OBJ files converted: {success_count}")
    print(f"OBJ files failed: {error_count}")
    print(f"XML files updated: {xml_updated}")
    print(f"XML files skipped: {xml_skipped}")
    print(f"Backups saved to: {backup_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
