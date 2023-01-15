#!/usr/bin/env python
import click
import os


@click.group()
def cli():
    pass

@click.group()
def run():
    pass

@click.group()
def docker():
    pass

@click.group()
def train():
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

@click.command()
def small():
    """Train small model"""
    click.echo('Trainning small model')
    from dnn.train_small import train_small as train_small
    os.chdir("./dnn")
    train_small()
    

run.add_command(streamlit)
run.add_command(fastapi)
run.add_command(all)

train.add_command(small)

cli.add_command(run)
cli.add_command(docker)
cli.add_command(train)

if __name__ == '__main__':
    cli()