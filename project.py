#!/usr/bin/env python
import click


@click.group()
def cli():
    pass

@click.group()
def run():
    pass

@click.group()
def docker():
    pass


@click.command()
def streamlit():
    """Run StreamLit data exploration GUI"""
    click.echo('Running StreamLit data exploration GUI')

@click.command()
def fastapi():
    """Run FastAPI microservice"""
    click.echo('Running FastAPI microservice')

@click.command()
def all():
    """Run all services"""
    click.echo('Running all services')
    

run.add_command(streamlit)
run.add_command(fastapi)
run.add_command(all)

cli.add_command(run)
cli.add_command(docker)

if __name__ == '__main__':
    cli()