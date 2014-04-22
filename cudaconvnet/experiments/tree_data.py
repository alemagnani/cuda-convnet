import json

# Simple caching
def cache(function):
    cache = {}

    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            rv = function(*args)
            cache[args] = rv
        return rv

    return wrapper

class TreeNodeData(object):
    def __init__(self,tree_filename,id_to_node=None):
        self.id_to_node = {}
        self.id_to_depth = {} #cache of depth
        if tree_filename is not None:
            with open(tree_filename, 'rb') as inputfile:
                for line in inputfile:
                    type = json.loads(line)
                    id = type.get('id')
                    parentId = type.get('parentId')
                    if parentId is not None:
                        if len(parentId) == 0:
                            type['parentId'] = None
                    if not id:
                        raise Exception('problem with id in row %s' % line)
                    self.id_to_node[id] = type
        elif id_to_node is not None:
            self.id_to_node = id_to_node
        else:
            raise Exception("cannot have both filename and id to none empty")

    def __iter__(self):
        return self.id_to_node.itervalues()

    def size(self):
        return len(self.id_to_node)

    def get_tree_node(self, id):
        return self.id_to_node.get(id)

    def get_tree_node_name(self,id):
        tmp = self.get_tree_node(id)
        if tmp is None:
            print 'problem of getting tree node name of node %s' % id
            return None
        else:
            return tmp.get('name')

    def get_path_name(self, tree_node_id):
        path = self.get_path(tree_node_id)
        return [self.get_tree_node(id).get('name') for id in path]

    def find_root_ids(self):

        out = []
        for node in self:
            parentId = node.get('parentId')
            if parentId is None:
                out.append(node.get('id'))
        return out


    def get_path(self,tree_node_id):
        path = []
        path.append(tree_node_id)
        node = self.get_tree_node(tree_node_id)
        if node is None:
            raise Exception('the id %s was not found' % tree_node_id)
        parentId = node.get('parentId')
        while parentId is not None :
            path.insert(0,parentId)
            node = self.get_tree_node(parentId)
            if not node:
                raise Exception('the id %s was not found' % parentId)
            parentId = node.get('parentId')
        return path

    @cache
    def get_children_ids(self, tree_node_id):
        #inefficient code TODO
        out = []
        for node in self:
            parentId = node.get('parentId')
            if parentId is not None and parentId == tree_node_id:
                out.append(node.get('id'))
        return out


    def get_all_children_ids(self, tree_node_id):
        out = []
        for children_id in self.get_children_ids(tree_node_id):
            out.append(children_id)
            out.extend(self.get_all_children_ids(children_id))
        return out

    def get_depth(self,tree_node_id):
        depth = self.id_to_depth.get(tree_node_id)
        if depth:
            return depth
        depth = 0
        node = self.get_tree_node(tree_node_id)
        if not node:
            raise Exception('the id %s was not found' % id)
        parentId = node.get('parentId')
        while parentId:
            depth += 1
            node = self.get_tree_node(parentId)
            if not node:
                raise Exception('the id %s was not found' % parentId)
            parentId = node.get('parentId')
        self.id_to_depth[tree_node_id] = depth
        return depth

    def l0_transform(self, y):
        out = []
        for id in y:
            try:
                l0 = self.get_path_name(id)[0]
            except:
                print 'cannot find id {}'.format(id)
                l0 = 'unknown'
            out.append(l0)
        return out
