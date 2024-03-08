all: book git

book:
	jupyter-book clean . --all
	jupyter-book build .
	
git:
	git add .
	git commit -m "update"
	git push
	ghp-import -n -p -f _build/html

clean:
	jupyter-book clean . --all
