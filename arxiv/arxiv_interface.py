import arxiv
import requests
import click

@click.command()
@click.argument('out_dir', type=click.Path())
@click.argument('query', default="tokamak", type=click.STRING)
@click.argument('n_papers', default=10, type=click.INT)
def run(out_dir, query, n_papers):
    search = arxiv.Search(
        query=query,
        max_results=n_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for res in search.results():
        path = res.download_pdf(out_dir)

if __name__ == '__main__':
    run()