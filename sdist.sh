#!/usr/bin/env bash

# Generate the distribution for upload to PyPI

rm dist/*

if python setup.py sdist bdist_wheel; then
    printf "\nᶘ ᵒᴥᵒᶅ  Build succeeded (in ./dist)\n\n"
else
    printf "\n〴⋋_⋌〵 Build failed\n\n"
    exit 1
fi

echo "Running \"twine check dist/*\""

if twine check dist/*; then
    printf "\n┌(☆o★)┘  twine check succeeded!\n\n"
else
    printf "\n(ʘдʘ╬)   twine check failed!\n\n"
    exit 1
fi

while true; do
    read -p "Do you wish to run \"twine upload dist/*\"? " yn
    case $yn in
        [Yy]* ) echo "Uploading..."; twine upload dist/*; exit ;;
        [Nn]* ) exit ;;
        * ) echo "Please answer yes or no.";;
    esac
done

