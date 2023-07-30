from src.opencone.opencone import OpenCone
import random
import numpy as np
import time
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

host = '<your domain>.<region>.es.amazonaws.com' # cluster endpoint, for example: my-test-domain.us-east-1.es.amazonaws.com
port = 443
region = '<region>'

credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region)


client = OpenSearch(
    hosts = [f'{host}:{port}'],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class=RequestsHttpConnection
)


oc = OpenCone(client=client)

# Create an index
dimensions = 1024 # OpenSearch 2.7 for KNN with filtering only supports 1024 dimensions
t = time.time()
try:
    oc.create_index(index_name="test", dimensions=dimensions, engine='lucene', space_type='cosinesimil')
except Exception as ex:
    if "resource_already_exists_exception" in str(ex):
        oc.delete_index(index_name="test")
        oc.create_index(index_name="test", dimensions=dimensions, engine='lucene', space_type='cosinesimil')
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
    "tags": {"$in": ["p1", "t3"]}
}
t = time.time()
rs = oc.search(index_name="test", vector=(np.random.randint(5, size=dimensions)/1.0).tolist(), filters=filters, metadata=False, limit=10000)
print("Search time:", time.time()-t)
for each in rs:
    print(each)

# Delete index
oc.delete_index(index_name="test")
