all:
	jupyter-book build .

push:
	ghp-import -n -p -f _build/html

clean:
	jupyter-book clean . --all
