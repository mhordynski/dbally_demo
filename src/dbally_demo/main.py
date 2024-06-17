import asyncio
import datetime
from typing import Annotated

import dbally
import sqlalchemy
from dbally import SqlAlchemyBaseView
from dbally.audit import CLIEventHandler
from dbally.embeddings import LiteLLMEmbeddingClient
from dbally.gradio import create_gradio_interface
from dbally.llms import LiteLLM
from dbally.similarity import SimilarityIndex, SimpleSqlAlchemyFetcher, FaissStore
from dbally.views import decorators
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base

engine = create_engine('sqlite:///clients.db')
load_dotenv()

Base = automap_base()
Base.prepare(autoload_with=engine)
Clients = Base.classes.clients


CityIndex = SimilarityIndex(
    fetcher=SimpleSqlAlchemyFetcher(sqlalchemy_engine=engine, table=Clients, column=Clients.city),
    store=FaissStore(
        index_dir="./similarity_indexes",
        index_name="country_similarity",
        embedding_client=LiteLLMEmbeddingClient(model="text-embedding-3-small"),
    )
)


class ClientsView(SqlAlchemyBaseView):

    def get_select(self) -> sqlalchemy.Select:
        return sqlalchemy.select(Clients)

    @decorators.view_filter()
    def filter_by_city(self, city: Annotated[str, CityIndex]):
        return Clients.city == city

    @decorators.view_filter()
    def eligible_for_loyalty_program(self):
        total_orders_check = Clients.total_orders > 3
        date_joined_check = Clients.date_joined < (datetime.datetime.now() - datetime.timedelta(days=365))

        return total_orders_check & date_joined_check


async def main():
    collection = dbally.create_collection("clients", llm=LiteLLM(model_name="gpt-4o"),
                                          event_handlers=[CLIEventHandler()])
    collection.add(ClientsView, lambda: ClientsView(engine))

    await collection.update_similarity_indexes()

    interface = await create_gradio_interface(collection)
    interface.launch()


if __name__ == '__main__':
    asyncio.run(main())
