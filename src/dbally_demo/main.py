import asyncio

import dbally
import sqlalchemy
from dbally import SqlAlchemyBaseView
from dbally.audit import CLIEventHandler
from dbally.gradio import create_gradio_interface
from dbally.llms import LiteLLM
from dbally.views import decorators
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base

engine = create_engine('sqlite:///clients.db')
load_dotenv()

Base = automap_base()
Base.prepare(autoload_with=engine)
Clients = Base.classes.clients


class ClientsView(SqlAlchemyBaseView):

    def get_select(self) -> sqlalchemy.Select:
        return sqlalchemy.select(Clients)

    @decorators.view_filter()
    def filter_by_city(self, city: str):
        return Clients.city == city


async def main():
    collection = dbally.create_collection("clients", llm=LiteLLM(model_name="gpt-4o"),
                                          event_handlers=[CLIEventHandler()])
    collection.add(ClientsView, lambda: ClientsView(engine))

    interface = await create_gradio_interface(collection)
    interface.launch()


if __name__ == '__main__':
    asyncio.run(main())
