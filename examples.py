from opencone import OpenconeClient
import random
import numpy as np
import time
from opensearchpy import OpenSearch


host = 'lbv'
port = 9200
auth = ('admin', 'admin') # For testing only. Don't store credentials in code.

# Create an OpenSearch client
client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )

oc = OpenconeClient(client=client)

# Create an index
dimensions = 4096
t = time.time()
try:
    oc.create_index(index_name="test", dimensions=dimensions)
except Exception as ex:
    if "resource_already_exists_exception" in str(ex):
        oc.delete_index(index_name="test")
        oc.create_index(index_name="test", dimensions=dimensions)
    else:
        raise ex
print("Index created in ", time.time()-t)

# Upsert vectors
titles = ["pp", "qq", "rr", "ss", "tt"]
tags = ["p1", "p2", "p3", "p4", "p5"]
vectors = []
for i in range(100):
    vectors.append((
        "id:"+str(i),
        np.random.randint(5, size=dimensions).tolist(),
        {
            "name": titles[random.randint(0, 4)],
            "tags": tags[random.randint(0, 4):random.randint(0, 4)],
            "no": random.randint(2000, 3000)
        }
    ))
t = time.time()
oc.upsert(index_name="test", vectors=vectors)
print("Upsert time:", time.time()-t)

# Fetch vector
print(oc.fetch(index_name="test", _id="id:1"))

# Delete vector
print(oc.delete(index_name="test", _id="id:1"))


# Search
filters = {
    "name": {"$eq": "pp"},
    "tags": {"$in": ["p1", "t3"]},
    "no":{"$gte":2200, "$lte":2800}
}
t = time.time()
rs = oc.search(index_name="test", vector=(np.random.randint(5, size=dimensions)/1.1).tolist(), filters=filters, metadata=False, limit=10000)

print("Search time:", time.time()-t)
for each in rs:
    print(each)

# Delete index
oc.delete_index(index_name="test")
