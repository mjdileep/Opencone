from typing import List, Tuple


class OpenconeClient:
    """
    OpenSearch wrapper for vector databases 
    """
    def __init__(self, client):
        """
        :param client: OpenSearch Client
        """
        self.client = client

    def create_index(self, index_name: str, dimensions: int, engine='faiss', method="hnsw", space_type='innerproduct'):
        """
        Creates a KNN index
        :param index_name: Name of the index
        :param dimensions: Dimensions
        :param engine: Matching engine
        :param method: Matching algorithm
        :param space_type: Space type (l2/innerproduct)
        :return:
        """
        body = {
            "settings": {
                "index": {"knn": True},
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimensions,
                        "method": {
                          "name": method,
                          "space_type": space_type,
                          "engine": engine
                        }

                    },
                }
            },
        }
        return self.client.indices.create(index_name, body=body)

    def delete_index(self, index_name: str):
        """
        Deletes a given index by name
        :param index_name: Name of the index
        :return:
        """
        return self.client.indices.delete(index=index_name)

    def upsert(self, index_name: str, vectors: List[Tuple]):
        """
        Insert/Update vectors as bulk (Recommends maximum 100 at a time)
        :param index_name: Index name
        :param vectors: List of [(id, embedding, metadata),...]
        :return:
        """
        body = []
        for _id, embeddings, metadata in vectors:
            body.append({"index": {"_index": index_name, "_id": _id}})
            metadata.update({"embedding": embeddings, "id": _id})
            body.append(metadata)
        self.client.bulk(index=index_name, body=body)
        return self.client.indices.refresh(index=index_name)

    def delete(self, index_name: str, _id: str):
        """
        Deletes vector by name
        :param index_name: Index name
        :param _id: ID of the vector
        :return:
        """
        response = self.client.delete( index=index_name, id=_id)
        return response

    def fetch(self, index_name: str, _id: str):
        """
        Fetch vector by name
        :param index_name: Index name
        :param _id: ID of the vector
        :return:
        """
        response = self.client.get(index=index_name, id=_id)
        return response["_source"]

    def __prepare_filters(self, filters):
        """
        {"genre":"comedy"} -> must  term - Done
        {"genre": {"$in":["documentary","action"]}} must in with support 1 term_set - Done
        {"$and": [{"genre": "comedy"}, {"genre":"documentary"}]} must in with support n term_set - done
        {"genre": {"$ne":comedy}} must not term
        {"genre": {"$eq":comedy}} -> must term - Done
        {"genre": {"$gt":comedy}} -> must range - Done
        {"genre": {"$gte":comedy}} -> must range - Done
        {"genre": {"$gte":comedy, "$lte":"erqr"}} -> must range - Done
        {"genre": {"$nin":["comady","ddd"]}} must not in with support 1 term_set - Done
        :param filters:
        :return:
        """
        filter_expression = {
            "must": [],
            "must_not": []
        }
        for k in filters:
            if k == "$and":
                term_sets = {}
                for elm in filters[k]:
                    t, v = elm.popitem()
                    if t in term_sets:
                        term_sets[t].append(v)
                    else:
                        term_sets[t] = [v]
                for term_set in term_sets:
                    filter_expression["must"].append({
                        "terms_set": {
                          term_set: {
                              "terms": term_sets[term_set],
                              "minimum_should_match_script": {
                                  "source": "Math.min(params.num_terms, {})".format(len(term_sets[term_set]))
                              }
                          }
                        }
                    })
            elif type(filters[k]) != dict:
                filter_expression["must"].append({
                        "term": {
                            k: {
                                "value": filters[k]
                            }
                        }

                    }
                )
            elif type(filters[k]) == dict:
                for kk in filters[k]:
                    if kk == "$in":
                        filter_expression["must"].append({
                            "terms_set": {
                                  k: {
                                      "terms": filters[k][kk],
                                      "minimum_should_match_script": {
                                          "source": "Math.min(params.num_terms, 1)"
                                      }
                                  }
                            }
                        })
                    elif kk in ("$gt", "$gte", "$lt", "$lte"):
                        is_new = True
                        for rule in filter_expression["must"]:
                            if rule.get("range") and rule["range"].get(k):
                                rule["range"][k].update({
                                    kk[1:]: filters[k][kk]
                                })
                                is_new = False
                                break
                        if is_new:
                            filter_expression["must"].append({
                                "range": {
                                      k: {
                                            kk[1:]: filters[k][kk]
                                      }
                                }
                            })
                    elif kk == "$eq":
                        filter_expression["must"].append({
                            "term": {
                                    k: {
                                        "value": filters[k][kk]
                                    }
                                }

                            }
                        )
                    elif kk == "$neq":
                        filter_expression["must_not"].append({
                            "term": {
                                    k: {
                                        "value": filters[k][kk]
                                    }
                                }

                            }
                        )
                    elif kk == "$nin":
                        filter_expression["must_not"].append({
                            "terms_set": {
                                  k: {
                                      "terms": filters[k][kk],
                                      "minimum_should_match_script": {
                                          "source": "Math.min(params.num_terms, 1)"
                                      }
                                  }
                            }
                        })
        if not filter_expression["must"]:
            filter_expression.pop("must")
        if not filter_expression["must_not"]:
            filter_expression.pop("must_not")
        return filter_expression

    def search(self, index_name: str, vector: List[float], filters: dict, metadata: bool = False, limit: int = 10):
        """
        Query vectors based on dotproduct/cosine(if vectors are normalised) similarity
        :param index_name: Name of the index
        :param vector: The vector to compare against
        :param filters: Filters if any
        :param metadata: Make True if you want to retrieve the metadata too (will be slow)
        :param limit: Max number of results
        :return: A list of matches sorted based on the score.
        """
        filters = self.__prepare_filters(filters)
        results = self.client.search(
            index=index_name,
            body={
                "size": limit,
                "sort": "_score",
                "query": {
                    "knn":
                        {
                            "embedding": {
                                "vector": vector,
                                "k": limit,
                                "filter": {
                                    "bool": filters
                                }
                            }
                        }
                },
                '_source': [] if metadata else ["$no_meta"]
            },
        )
        return results["hits"]["hits"]


