from .common import list_to_dict
from .common import load_from_yaml_file
from .common import gen_uuid
from pymongo import MongoClient
import pymongo
import copy
from bson import ObjectId
from datetime import datetime
from collections import OrderedDict
from collections import defaultdict
import logging
from tqdm import tqdm
from .common import print_table
from .common import try_once

def create_mongodb_client():
    config = load_from_yaml_file('./aux_data/configs/mongodb_credential.yaml')
    host = config['host']
    return MongoClient(host)

def create_bbverification_db(db_name='qd', collection_name='uhrs_bounding_box_verification'):
    '''
    use create_bbverificationdb_client since the naming is not precise
    '''
    return BoundingBoxVerificationDB(db_name, collection_name)

def create_bbverificationdb_client(db_name='qd',
        collection_name='uhrs_bounding_box_verification'):
    return BoundingBoxVerificationDB(db_name, collection_name)

def objectid_to_str(result):
    # convert the type of ObjectId() to string
    result = list(result)
    for r in result:
        r['_id'] = str(r['_id'])
    return result

def ensure_objectid(result):
    for r in result:
        if type(r['_id']) is str:
            r['_id'] = ObjectId(r['_id'])

def ensure_to_objectid(r):
    if type(r) is str:
        return ObjectId(r)
    else:
        return r

def create_annotation_db():
    return AnnotationDB()

class AnnotationDB(object):
    '''
    gradually move all the Annotation db related function call to this wrapper.
    The related table incldues: image, ground_truth, label, prediction_result
    '''
    def __init__(self):
        self._qd = create_mongodb_client()
        self._gt = self._qd['qd']['ground_truth']
        self._label = self._qd['qd']['label']
        self._acc = self._qd['qd']['acc']
        self._phillyjob = self._qd['qd']['phillyjob']
        self._cluster = self._qd['qd']['cluster']
        import getpass
        self._judge = self._qd['qd']['judge']
        self.username = getpass.getuser()

    def add_meta_data(self, kwargs):
        if 'create_time' not in kwargs:
            kwargs['create_time'] = datetime.now()
        if 'username' not in kwargs:
            kwargs['username'] = self.username
        kwargs['update_time'] = datetime.now()

    def insert_judge(self, **kwargs):
        self.add_meta_data(kwargs)
        self._judge.insert_one(kwargs)

    def insert_cluster_summary(self, **kwargs):
        self.add_meta_data(kwargs)
        self._cluster.insert_one(kwargs)

    def insert_phillyjob(self, **kwargs):
        # use self.add_meta_data
        self.add_meta_data(kwargs)
        self._phillyjob.insert_one(kwargs)

    def insert_acc(self, **kwargs):
        self.add_meta_data(kwargs)
        self._acc.insert_one(kwargs)

    def insert_label(self, **kwargs):
        if 'uuid' not in kwargs:
            kwargs['uuid'] = gen_uuid()
        if 'create_time' not in kwargs:
            kwargs['create_time'] = datetime.now()

        self._label.insert_one(kwargs)

    def insert_one(self, collection_name, **kwargs):
        self.add_meta_data(kwargs)
        result = self._qd['qd'][collection_name].insert_one(kwargs)
        return result

    def remove_phillyjob(self, **kwargs):
        self._phillyjob.delete_many(kwargs)

    def delete_many(self, collection_name, **kwargs):
        self._qd['qd'][collection_name].delete_many(kwargs)

    def update_one(self, doc_name, query, update, **kwargs):
        return self._qd['qd'][doc_name].update_one(
                query,
                update,
                **kwargs
                )

    def update_many(self, doc_name, query, update):
        if '$set' in update:
            update['$set']['update_time'] = datetime.now()
        return self._qd['qd'][doc_name].update_many(query, update)

    def update_phillyjob(self, query, update):
        return self._phillyjob.update_many(query, {'$set': update})

    def iter_judge(self, **kwargs):
        return self._judge.find(kwargs)

    def iter_phillyjob(self, **kwargs):
        return self._phillyjob.find(kwargs)

    def iter_general(self, table_name, **kwargs):
        return self._qd['qd'][table_name].find(kwargs).sort('create_time', -1)

    def iter_acc(self, **query):
        return self._acc.find(query)

    def update_one_acc(self, query, update):
        self._acc.update_one(query, update)

    def iter_unique_test_info_in_acc(self):
        pipeline = [
                {'$group': {'_id': {'test_data': '$test_data',
                                    'test_split': '$test_split',
                                    'test_version': '$test_version'}}}
                ]
        for result in self._acc.aggregate(pipeline):
            yield result['_id']

    def exist_acc(self, **query):
        try:
            next(self.iter_acc(**query))
            return True
        except:
            return False

    # label related
    def update_label(self, query, update):
        self._label.update_one(query, update)

    def iter_label(self):
        return self._label.find()

    def iter_query_label(self, query):
        return self._label.find(query)

    def build_label_index(self):
        self._label.create_index([('uuid', 1)], unique=True)
        self._label.create_index([('unique_name', 1)], unique=True,
                collation={'locale': 'en', 'strength':2})

    def create_index(self, collection, *args, **kwargs):
        self._qd['qd'][collection].create_index(*args, **kwargs)

    def drop_ground_truth_index(self):
        self._gt.drop_indexes()

    def build_job_index(self):
        self._phillyjob.create_index([('create_time', 1)])
        self._phillyjob.create_index([('appID', 1)], unique=True)

    def build_ground_truth_index(self):
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('key', 1),
            ('class', 1)])
        self._gt.create_indexes
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('class', 1)])
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('class', 1),
            ('version', 1)])
        # used for deleting all before inserting
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('version', 1)])

class BoundingBoxVerificationDB(object):
    status_requested = 'requested'
    status_retrieved = 'retrieved'
    status_submitted = 'submitted'
    status_completed = 'completed'
    status_merged = 'merged'
    urgent_priority_tier = -10000
    def __init__(self, db_name='qd', collection_name='uhrs_bounding_box_verification'):
        self.client = None
        self.db_name = db_name
        self.collection_name = collection_name

    def query_by_pipeline(self, pipeline):
        result = self.collection.aggregate(pipeline, allowDiskUse=True)
        return list(result)

    def query_verified_correct_rects(self, data, split, key):
        pipeline = [{'$match': {'data': data,
                                'split': split,
                                'key': key,
                                'interpretation_result': 1,
                                'status': {'$in': [self.status_completed,
                                                   self.status_merged]}}},
                    ]
        return [info['rect'] for info in self.query_by_pipeline(pipeline)]

    def query_verified_incorrect_rects(self, data, split, key):
        pipeline = [{'$match': {'data': data,
                                'split': split,
                                'key': key,
                                'interpretation_result': {'$ne': 1},
                                'status': {'$in': [self.status_completed,
                                                   self.status_merged]}}},
                    ]
        return [info['rect'] for info in self.query_by_pipeline(pipeline)]

    def query_nonverified_rects(self, data, split, key):
        pipeline = [{'$match': {'data': data,
                                'split': split,
                                'key': key,
                                'status': {'$nin': [self.status_completed,
                                                   self.status_merged]}}},
                    ]
        result = []
        for info in self.query_by_pipeline(pipeline):
            info['rect']['_id'] = str(info['_id'])
            result.append(info['rect'])
        return result

    def request_by_insert(self, all_box_task):
        def get_bb_task_id(rect_info):
            from .common import hash_sha1
            rect = rect_info['rect']
            return hash_sha1([rect_info['url'], rect['class'], rect['rect']])
        all_box_task = copy.deepcopy(all_box_task)
        for b in all_box_task:
            assert 'status' not in b
            assert 'priority_tier' in b, 'priority' in b
            assert 'url' in b
            assert 'rect' in b
            b['status'] = self.status_requested
            b['last_update_time'] = {'last_{}'.format(self.status_requested):
                    datetime.now()}
            if 'rect' not in b:
                b['rect'] = b['rects'][0]
            b['bb_task_id'] = get_bb_task_id(b)
        self.collection.insert_many(all_box_task)

    def retrieve(self, max_box, urgent_task=False):
        assert max_box > 0
        sort_config = OrderedDict()
        sort_config['priority_tier'] = pymongo.ASCENDING
        sort_config['priority'] = pymongo.ASCENDING
        match_criteria = {'status': self.status_requested}
        if urgent_task:
            match_criteria['priority_tier'] = self.urgent_priority_tier
        pipeline = [
                {'$match': match_criteria},
                {'$sort': sort_config},
                {'$limit': max_box},
                ]
        result = self.query_by_pipeline(pipeline)
        # we need to update the status to status_retrieved to avoid duplicate
        # retrieve && submit
        self.update_status([r['_id'] for r in result],
                self.status_retrieved)
        return objectid_to_str(result)

    def update_priority_tier(self, all_id, new_priority_tier):
        all_id = [ensure_to_objectid(i) for i in all_id]
        self.collection.update_many({'_id': {'$in': all_id}},
                                    {'$set': {'priority_tier': new_priority_tier}})

    def update_status(self, all_id, new_status, allowed_original_statuses=None):
        all_id = list(set(all_id))
        for i in range(len(all_id)):
            if type(all_id[i]) is str:
                all_id[i] = ObjectId(all_id[i])
        query = {'_id': {'$in': all_id}}
        if allowed_original_statuses:
            query['status'] = {'$in': allowed_original_statuses}
        time_key = 'last_update_time.last_{}'.format(new_status)
        result = self.collection.update_many(filter=query,
                update={'$set': {'status': new_status,
                                 time_key: datetime.now()}})

    def reset_status_to_requested(self, all_bb_task):
        self.update_status([b['_id'] for b in all_bb_task],
                self.status_requested)

    def submitted(self, submitted):
        self.adjust_status(submitted, 'uhrs_submitted_result',
                'uhrs_submitted_result', self.status_submitted)

    def adjust_status(self, uhrs_results, uhrs_result_field, db_field,
            new_status, allowed_original_statuses=None):
        uhrs_results = list(uhrs_results)
        for s in uhrs_results:
            assert uhrs_result_field in s
            assert '_id' in s

        ensure_objectid(uhrs_results)
        all_id = [s['_id'] for s in uhrs_results]

        # save the result from uhrs
        for s in uhrs_results:
            self.collection.update_one(filter={'_id': s['_id']},
                    update={'$set': {db_field: s[uhrs_result_field]}})

        # update the status
        self.update_status(all_id, new_status, allowed_original_statuses)

    def query_submitted(self, topk=None):
        pipeline = [
                {'$match': {'status': self.status_submitted}},
                ]
        if topk:
            pipeline.append({'$limit': topk})
        result = self.query_by_pipeline(pipeline)
        return objectid_to_str(result)

    def complete(self, completed):
        self.adjust_status(completed, 'uhrs_completed_result',
                'uhrs_completed_result',
                self.status_completed,
                allowed_original_statuses=[self.status_submitted])

    def set_status_as_merged(self, all_id):
        self.update_status(all_id, self.status_merged,
                allowed_original_statuses=[self.status_completed])

    def get_completed_uhrs_result(self, extra_match=None):
        merge_multiple_verification = False # True if we submit one rect multiple times, not tested
        match_criteria = {'status': self.status_completed}
        if extra_match:
            match_criteria.update(extra_match)

        pipeline = [
                {'$match': match_criteria},
                ]

        if merge_multiple_verification:
            pipeline.append(
                {'$group': {'_id': {'data': '$data',
                                    'split': '$split',
                                    'key': '$key',
                                    'bb_task_id': '$bb_task_id'},
                            'rects': {'$first': '$rects'},
                            'uhrs': {'$push': '$uhrs_completed_result'},
                            'related_ids': {'$push': '$_id'},
                            }}
                    )
        data_split_to_key_rects = defaultdict(list)
        all_id = []
        logging.info('querying the completed tasks')
        for rect_info in tqdm(self.query_by_pipeline(pipeline)):
            data = rect_info['data']
            split = rect_info['split']
            rect = rect_info['rect']
            all_id.append(rect_info['_id'])
            rect['uhrs'] = rect_info['uhrs_completed_result']
            key = rect_info['key']
            data_split_to_key_rects[(data, split)].append((key, rect))
        return data_split_to_key_rects, all_id

    @property
    def collection(self):
        if self.client is None:
            self.client = create_mongodb_client()
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('split', pymongo.ASCENDING),
                ('key', pymongo.ASCENDING)])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rects.0.class', pymongo.ASCENDING),
                ('rects.0.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rect.class', pymongo.ASCENDING),
                ('rect.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rects.0.class', pymongo.ASCENDING),
                ('rects.0.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ('priority', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rect.class', pymongo.ASCENDING),
                ('rect.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ('priority', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING)])
            self.collection.create_index([('priority_tier', pymongo.ASCENDING)])
            self.collection.create_index([('status', pymongo.ASCENDING)])
            self.collection.create_index([('priority', pymongo.ASCENDING)])
            self.collection.create_index([('rects.0.from', pymongo.ASCENDING)])
            self.collection.create_index([('rect.from', pymongo.ASCENDING)])
        return self.client[self.db_name][self.collection_name]

def inject_cluster_summary(info):
    c = create_annotation_db()
    c.insert_cluster_summary(**info)

def update_cluster_job_db(all_job_info, collection_name='phillyjob'):
    c = create_annotation_db()
    existing_job_infos = list(c.iter_general(collection_name))

    appID_to_record = {j['appID']: j for j in existing_job_infos}

    for job_info in all_job_info:
        non_value_keys = [k for k, v in job_info.items() if v is None]
        for k in non_value_keys:
            del job_info[k]
        if job_info['appID'] in appID_to_record:
            record = appID_to_record[job_info['appID']]
            need_update = False
            for k, v in job_info.items():
                if k in ['elapsedTime', 'elapsedFinished']:
                    continue
                if k == 'data_store':
                    v = sorted(v)
                    record[k] = sorted(v)
                from .common import float_tolorance_equal
                if k not in record or not float_tolorance_equal(record[k], v,
                        check_order=False):
                    need_update = True
                    logging.info('update because {} need to be changed from {}'
                            ' to {}'.format(k, record.get(k), v))
                    break
            if need_update:
                c.update_many(collection_name,
                        query={'appID': job_info['appID']},
                        update={'$set': job_info})
        else:
            try:
                c.insert_one(collection_name, **job_info)
            except:
                # if two instances are running to inject to db, there might be
                # a chance that a new job is inserted here at the same time.
                # For the db, we make the appID unique, and one of the
                # instances will fail. Thus, we just ignore the error here
                from .common import print_trace
                print_trace()

@try_once
def try_query_job_acc(*args, **kwargs):
    query_job_acc(*args, **kwargs)

def get_job_ids_by_scheduler_id(_ids):
    from qd.gpucluster.job_scheduler import JobScheduler
    c = create_annotation_db()
    scheduler_info = [job_info for job_info in c.iter_general(
        JobScheduler.collection,
        _id={'$in': _ids},
    )]
    _id_to_scheduler = {s['_id']: s for s in scheduler_info}
    return [{'scheduler_id': _id, 'appID': _id_to_scheduler[_id].get('appID')} for _id in _ids]

def query_job_acc(job_ids, key='appID', inject=False,
                  must_have_any_in_predict=None,
                  not_have_any_in_predict=None,
                  metric_names=None):
    if all(isinstance(i, ObjectId) for i in job_ids):
        job_infos = get_job_ids_by_scheduler_id(job_ids)
    else:
        job_infos = job_ids
    c = create_annotation_db()
    query_param = {key: {'$in': [j['appID'] for j in job_infos]}}
    jobs = list(c.iter_phillyjob(**query_param))
    appID_to_info = {j['appID']: j for j in job_infos}
    for j in jobs:
        j.update(appID_to_info[j['appID']])
    for j in jobs:
        if all(k in j for k in ['data', 'net', 'expid']):
            j['full_expid'] = '_'.join([str(j[k]) for k in ['data', 'net', 'expid']])
    all_full_expid = []

    for j in jobs:
        if 'data' not in j or 'net' not in j or 'expid' not in j:
            full_expid = 'U'
        else:
            full_expid = '_'.join(map(str, [j.get('data', ''), j.get('net',''),
                                            j.get('expid', '')]))
        all_full_expid.append(full_expid)
    all_key = ['scheduler_id', 'appID', 'status', 'result', 'speed', 'left', 'eta',
        'full_expid', 'num_gpu', 'mem_used', 'gpu_util']
    all_key = [k for k in all_key if any(j.get(k) is not None for j in jobs)]

    #print_table([j for j in jobs if j['status'] == 'Failed'],  all_key=all_key)
    print_table(jobs,  all_key=all_key)
    if len(all_full_expid) == 0:
        print_table(job_ids)

    if inject:
        for full_expid in all_full_expid:
            from qd.process_tsv import inject_accuracy_one
            inject_accuracy_one(full_expid)

    query_acc_by_full_expid(all_full_expid,
                            must_have_any_in_predict=must_have_any_in_predict,
                            not_have_any_in_predict=not_have_any_in_predict,
                            metric_names=metric_names)

def query_acc_by_full_expid(all_full_expid,
                            must_have_any_in_predict=None,
                            not_have_any_in_predict=None,
                            metric_names=None,
                            sort_row=True,
                            ):
    c = create_annotation_db()
    acc = list(c.iter_acc(full_expid={'$in': all_full_expid}))
    full_expid_to_rank = {f: i for i, f in enumerate(all_full_expid)}
    if not sort_row:
        acc = sorted(acc, key=lambda x: full_expid_to_rank[x['full_expid']])
    if metric_names is None:
        metric_names = [
            'all-all',
            'overall$0.5$map',
            'overall$-1$map',
            'top1',
            'global_avg',
            'feat_eig_value_ratio',
            'feat_mean_value',
            'feat_eig_max_value_ratio',
            'Bleu_4',
            'METEOR',
            'ROUGE_L',
            'CIDEr',
            'SPICE',
            'attr',
            't2i$R@1',
            't2i$R@10',
            't2i$R@5',
            'i2t$R@1',
            'i2t$R@10',
            'i2t$R@5',
            'acc',
            'stvqa_acc',
            'stvqa_anls',
            'vqa_acc',
        ]
    acc = [a for a in acc if a['metric_name'] in metric_names]
    if must_have_any_in_predict is not None:
        acc = [a for a in acc if any(t in a['report_file'] for t in
            must_have_any_in_predict)]
    if not_have_any_in_predict is not None:
        acc = [a for a in acc if all(t not in a['report_file'] for t in
                not_have_any_in_predict)]
    for a in acc:
        del a['username']
        del a['_id']
        del a['create_time']
        a['metric_value'] = round(a['metric_value'], 3)
        del a['test_split']
        del a['test_version']
        if 'predict_file' in a:
            del a['predict_file']
        #del a['report_file']
        del a['test_data']
    from .common import natural_key
    if sort_row:
        acc = sorted(acc, key=lambda a: (
            natural_key(a['full_expid']),
            a['metric_value'],
            ))
    #from .common import remove_empty_keys_
    #remove_empty_keys_(acc)
    # group by metric names
    all_metric = list(set([a['metric_name'] for a in acc]))
    all_metric = sorted(all_metric)
    fp_to_acc = {}
    for a in acc:
        fp = (a['full_expid'], a['report_file'])
        if fp in fp_to_acc:
            fp_to_acc[fp].append(a)
        else:
            fp_to_acc[fp] = [a]
    all_x = []
    for fp, accs in fp_to_acc.items():
        x = {'full_expid': fp[0], 'report_file': fp[1]}
        for m in all_metric:
            x[m] = None
        for a in accs:
            x[a['metric_name']] = a['metric_value']
        all_x.append(x)

    all_key = ['full_expid']
    all_key.extend(all_metric)
    all_key.append('report_file')
    # special case for caption prod
    full_expid_to_infos = list_to_dict(
        [(x['full_expid'], x) for x in all_x], 0)
    all_extra = []
    all_remove = []
    merge_data = ['TaxGettyImagesClean.test', 'TaxUserInsertedClean.test', 'TaxCaptionBot.trainval']
    for full_expid, infos in full_expid_to_infos.items():
        for curr_info in infos:
            if merge_data[0] not in curr_info['report_file']:
                continue
            is_merge = True
            merge_infos = []
            for i in range(1, len(merge_data)):
                target = curr_info['report_file'].replace(merge_data[0], merge_data[i])
                founds = [_i for _i in infos if _i['report_file'] == target]
                if len(founds) == 0:
                    is_merge = False
                    break
                merge_infos.append(founds[0])
            if is_merge:
                merge_infos.append(curr_info)
                extra = {'full_expid': full_expid,
                         'report_file': curr_info['report_file'].replace(merge_data[0], 'avg'),
                         }
                for k in all_key[1:6]:
                    v = sum([i[k] for i in merge_infos]) / len(merge_infos)
                    extra[k] = round(v, 3)
                all_extra.append(extra)
                all_remove.extend(merge_infos)
    for r in all_remove:
        all_x.remove(r)
    all_x.extend(all_extra)
    remove_not_best_not_last_(all_x, all_metric)
    print_table(all_x, all_key=all_key)

def remove_not_best_not_last_(all_x, metric_keys):
    full_expid_to_infos = list_to_dict(
        [(x['full_expid'], x) for x in all_x], 0)
    to_remove = []
    for full_expid, infos in full_expid_to_infos.items():
        for i in infos:
            assert i['report_file'][10] == '_'
            i['_iter'] = int(i['report_file'][11:18])
            left = i['report_file'][18:]
            assert i['report_file'][18] == '.'
            i['_left'] = left
        left_to_infos = list_to_dict([(i['_left'], i) for i in infos], 0)
        for left, curr_infos in left_to_infos.items():
            max_iter = max([i['_iter'] for i in curr_infos])
            max_metric = [max([i[mk] for i in curr_infos]) for mk in
                          metric_keys]
            for i in curr_infos:
                if i['_iter'] != max_iter and \
                        all(i[k] != max_k for k, max_k in zip(metric_keys, max_metric)):
                    to_remove.append(i)
    for t in to_remove:
        all_x.remove(t)
    for x in all_x:
        del x['_iter']
        del x['_left']




