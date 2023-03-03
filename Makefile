all:
	jupyter-book clean . --all
	jupyter-book build .
	git add .
	git commit -m "update"
	git push
	ghp-import -n -p -f _build/html

clean:
	jupyter-book clean . --all
