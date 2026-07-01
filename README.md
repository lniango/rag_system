# RAG turtorial

# Key components of a RAG system
# Document loaders, text splitting, and indexing
- *Document Loaders*: These tools pull in data from various sources (text files, PDFs, databases…) They convert that info into a format the system can actually use. Basically, we make sure all the important data is ready and in the right shape for the next steps.
- *Text splitting*: Once the data is loaded, it gets chopped into smaller chunks. This is super important because smaller pieces are easier to search through, and language models work better with bite-sized bits of info due to their processing limits.
- *Indexing*: After splitting, you need to organize the data. Indexing turns those text chunks into vector representations. This setup makes it easy and fast for the system to search through all that data and find what’s most relevant to a user’s query.

# Retrieval modes
These are the heart of the RAG system. They’re responsible for digging through all that indexed data to find what you need.

- *Vector Stores*: These are databases designed to handle those vector representations of the text chunks. They make searching super efficient by using a method called vector similarity search, which compares the query to the stored vectors and pulls the best matches.

- *Retrievers*: These components do the actual searching. They take the user’s query, convert it into a vector, and then search the vector store to find the most relevant data. Once they grab that info, it’s passed along to the next step: generation.

# Generative models
Now, this is where the magic happens. Once the relevant data is retrieved, the generative models take over and produce a final response.

- *Language Models*: These models create the actual response, making sure it’s coherent and fits the context. In a RAG system, they take both the retrieved data and their own internal knowledge to generate a response that’s up-to-date and accurate.

- *Contextual Response Generation*: The generative model blends the user’s question with the retrieved data to create a response that not only answers the question but also reflects the specific details from the relevant info it pulled.

