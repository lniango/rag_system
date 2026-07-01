# RAG turtorial

# Key components
# Document loaders, text splitting, and indexing
- *Document Loaders*: These tools pull in data from various sources (text files, PDFs, databases…) They convert that info into a format the system can actually use. Basically, we make sure all the important data is ready and in the right shape for the next steps.
- *Text splitting*: Once the data is loaded, it gets chopped into smaller chunks. This is super important because smaller pieces are easier to search through, and language models work better with bite-sized bits of info due to their processing limits.
- *Indexing*: After splitting, you need to organize the data. Indexing turns those text chunks into vector representations. This setup makes it easy and fast for the system to search through all that data and find what’s most relevant to a user’s query.