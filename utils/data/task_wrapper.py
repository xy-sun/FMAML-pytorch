import learn2learn as l2l


class TaskLoader:

    def __init__(self, dataset, n_way, k_shot, k_query, length, batch_size=None):
        self.dataset=dataset
        self.n_way= n_way
        self.k_shot=k_shot
        self.k_query= k_query
        self.length=length
        self.batch_size=batch_size
        if batch_size is None:
            self.generator = l2l.data.TaskGenerator(dataset, n_way, k_shot + k_query, tasks=length)
        else:
            self.generator = l2l.data.TaskGenerator(dataset, n_way, k_shot + k_query, tasks=length*batch_size)
        return

    def __iter__(self):
        def closure():
            if self.batch_size is None:
                for _ in range(self.length):
                    self.support_set = self.generator.sample(self.k_shot)
                    self.query_set = self.generator.sample(self.k_query,  self.support_set.sampled_task)
                    yield self.support_set, self.query_set
            else:
                for _ in range(self.length):
                    batch = []
                    for __ in range(self.batch_size):
                        self.support_set = self.generator.sample(self.k_shot)
                        self.query_set = self.generator.sample(self.k_query,  self.support_set.sampled_task)
                        batch.append((self.support_set, self.query_set))
                    yield batch
        return closure()



# def task_loader(dataset, n_way, k_shot, k_query, length, batch_size=None):
#     if batch_size is None:
#         generator = l2l.data.TaskGenerator(dataset, n_way, k_shot + k_query, tasks=length)
#         for _ in range(length):
#             support_set = generator.sample(k_shot)
#             query_set = generator.sample(k_query, support_set.sampled_task)
#             yield support_set, query_set
#         return
#     else:
#         generator = l2l.data.TaskGenerator(dataset, n_way, k_shot + k_query, tasks=length*batch_size)
#         for _ in range(length):
#             batch = []
#             for __ in range(batch_size):
#                 support_set = generator.sample(k_shot)
#                 query_set = generator.sample(k_query, support_set.sampled_task)
#                 batch.append((support_set, query_set))
#             yield batch
#         return
