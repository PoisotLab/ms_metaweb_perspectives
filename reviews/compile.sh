#find . -type f -name "*.md" -exec bash -c 'pandoc {} -o {}.pdf --template=../.assets/templates/responses.tex' \;
find . -type f -name "*.md" -exec bash -c 'pandoc {} -o {}.pdf' \;