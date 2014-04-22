from optparse import OptionParser
import os


def main():

    op = OptionParser()

    op.add_option("--output_folder", default='/Users/alessandro/Desktop/kaggle/stage',
                  action="store", type=str, dest="output_folder",
                  help="Train data.")
    op.add_option("--matrix_file", default='/Users/alessandro/Desktop/kaggle/train_matrix.p',
                  action="store", type=str, dest="matrix_file",
                  help="Train data output.")

    op.add_option("--hierarchy_file", default='/Users/alessandro/Desktop/kaggle/hierarchy.txt',
                  action="store", type=str, dest="hierarchy_file",
                  help="Train data output.")

    (opts, args) = op.parse_args()

    output_folder = opts.output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    tree_data = TreeData(opts.hierarchy_file)

    find_splits(tree_data)

def find_splits(tree_node_data, max=300):
    m = {}
    splits = []
    for root_id in tree_node_data.find_root_ids():
        m[root_id] = root_id
        find_split_recursive(root_id,tree_node_data,splits,max=max)
    splits.append((None,m))
    return splits

def find_split_recursive(id, tree_node_data, splits, max=300):

    children =  tree_node_data.get_all_children_ids(id)
    if len(children) ==0:
        return
    print 'finding splits for {}'.format(id)
    if len(children) < max:
        print 'stopping at id {} with size of children {}'.format(id, len(children))
        m = { child : child for child in children}
        m[id] = id
        splits.append((id, m))
        return
    else:
        print 'expanding node {} with {} children ----------------------------------------------------'.format(id,len(children))
        m = {id: id}
        for direct_child  in tree_node_data.get_children_ids(id):
            find_split_recursive(direct_child, tree_node_data, splits, max=max)
            m[direct_child] = direct_child
            for child_child in tree_node_data.get_all_children_ids(direct_child):
                m[child_child] = direct_child
        splits.append((id, m))

class TreeData():
    def __init__(self, file):
        self.all_ids ,self.child_to_parent, self.parent_to_children = read_classes(file)
        print 'total number of nodes {}'.format(len(self.all_ids))


    def find_root_ids(self):
        out = []
        for node in self.all_ids:
            if node not in self.child_to_parent:
                out.append(node)
        print 'root ids are of size {}'.format(len(out))
        return out

    def get_all_children_ids(self, id):
        out = []
        for child in self.get_children_ids(id):
            out.append(child)
            out.extend(self.get_all_children_ids(child))
        return out

    def get_children_ids(self,id):
        if id in self.parent_to_children:
            return [c for c in self.parent_to_children.get(id)]
        return []



def read_classes(file):
    child_to_parent = {}
    parent_to_children = {}
    all_ids = set()
    with open(file, 'rb') as f:
        for line in f:
            parts = line.split(' ')
            if parts is None or len(parts) == 0:
                continue
            assert (len(parts) == 2)
            parent = int(parts[0])
            child = int(parts[1])
            all_ids.add(parent)
            all_ids.add(child)
            if child in child_to_parent:
                print 'problem with data not being a tree {} {} {}'.format(child,parent, child_to_parent.get(child))
                raise Exception('problem with tree')
            child_to_parent[child] = parent
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children.get(parent).append(child)
    return all_ids, child_to_parent, parent_to_children



if __name__ == "__main__":
    main()