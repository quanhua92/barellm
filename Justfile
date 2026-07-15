run:
	uv run barellm run

serve:
	uv run barellm serve

pull:
	uv run barellm pull

ls:
	uv run barellm ls

rm model:
	uv run barellm rm {{model}}

test:
	uv run pytest

lint:
	uv run ruff check --fix .
	uv run ruff format .
