set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

run prompt *args:
	uv run barellm run "{{prompt}}" {{args}}

serve:
	uv run barellm serve

pull *args:
	uv run barellm pull {{args}}

ls:
	uv run barellm ls

rm model:
	uv run barellm rm {{model}}

test:
	uv run pytest

lint:
	uv run ruff check --fix .
	uv run ruff format .
