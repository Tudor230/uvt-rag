from llama_index.core import PromptTemplate

QUERY_GEN_STR =  """
Ești un asistent util care generează mai multe interogări de căutare pe baza unei singure interogări introduse de utilizator.
Generează 3 interogări de căutare în limba română, fiecare pe un rând separat, legate de următoarea interogare:
Interogare: {query}
Interogări:
"""

query_gen_prompt = PromptTemplate(QUERY_GEN_STR)