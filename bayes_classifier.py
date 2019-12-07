#%%
import numpy as np

#%%
def read_iris(filepath = r'..\..\Iris Dataset.txt'):
    '''读取文件中的属性值'''
    ans = []
    with open(filepath) as infile:
        for line in infile:
            tokens = line.strip().split(' ')
            ans.append([float(x) for x in tokens])
    return np.array(ans)
#%%
def cal_expection_variance(col_data):
    '''计算均值和标准差'''
    u = np.mean(col_data)
    variance = np.sqrt(np.mean(np.square(col_data-u)))
    return u, variance

def normal_distribution(x, u, deta):
    '''返回均值u，标准差deta，正态分布的值'''
    w = 1.0/(np.sqrt(2*np.pi)*deta)
    exponent = -0.5*(((x-u)/deta)**2)
    return w*np.exp(exponent)

class Normal_bayes_classifier:
    '''朴素贝叶斯的分类器'''
    
    def train_NBC(self, train_data, atrr_num, class_num):
        '''计算朴素贝叶斯所需要的参数, class_num 类的数目'''
        class_data = []
        for i in range(class_num):
            data = train_data[np.where(train_data[:, -1]==(1+i))]
            class_data.append(data)
        uv_set = np.zeros((class_num, atrr_num, 2))
        class_pro = np.zeros((class_num))
        for i in range(class_num):
            class_pro[i] = len(class_data[i])/len(train_data)
            for col in range(atrr_num):
                col_data = class_data[i][:, col]
                u,v = cal_expection_variance(col_data)
                uv_set[i][col] = u,v
        self.class_pro = class_pro
        self.uv_para = uv_set
        #print('train_finish!')
        #print('class_pro:{}'.format(self.class_pro))


    def predict_class(self, attr_vector):
        '''预测数据所属的类别'''
        attr_prod = np.ones((3))
        for c in range(3):
            for col in range(len(attr_vector)):
                nd = normal_distribution(attr_vector[col], self.uv_para[c][col][0], self.uv_para[c][col][1])
                attr_prod[c] *= nd
                #print('c:{} col:{} nd:{}'.format(c, col, nd))
        p_c_by_x = self.class_pro*attr_prod
        pre_c = np.argmax(p_c_by_x) + 1
        #print('p(c|x):{} predict:{}'.format(p_c_by_x, pre_c))
        return pre_c


#%%
class Flexiable_bayes_classfier:
    '''灵活贝叶斯分类器'''

    def train_FBC(self, train_data, class_num):
        '''训练灵活贝叶斯分类器'''
        self.class_data = []
        self.class_pro = np.zeros((class_num))
        for i in range(class_num):
            data = train_data[np.where(train_data[:, -1]==(1+i))]
            self.class_pro[i] = len(data)/len(train_data)
            self.class_data.append(data[:,:-1])

    def __cal_joint_probability(self, class_index, attr_vector):
        '''计算属性attr_vector 在class_index 下的联合概率'''
        c_data = self.class_data[class_index]
        N = len(c_data)
        h = 1.0/np.sqrt(N)
        ans = 0.0
        for x in c_data:
            coef = 1.0/(np.sqrt(2*np.pi)*h)
            p2 = np.exp(-0.5*np.square((x-attr_vector)/h))
            ans += np.prod(coef*p2)
        return ans

    def predict(self, attr_vector):
        '''预测类'''
        joint_pro = np.zeros((3))
        for i in range(3):
            joint_pro[i] = self.__cal_joint_probability(i, attr_vector)
        pre_c = np.argmax(self.class_pro*joint_pro) +1
        return pre_c



#%%
def split_data_to_ten(raw_data):
    '''返回分割为十份的数据'''
    t_data = [[] for i in range(10)] 
    for i in range(3):
        c_i = np.where(raw_data[:,-1]==(i+1))
        c_data = raw_data[c_i]
        for j in range(10):
            t_data[j].append(c_data[j*5:(j+1)*5])
    for i in range(10):
        t_data[i] = np.row_stack(t_data[i])
    return t_data
    


#%%
if __name__ == "__main__":
    iris_data = read_iris()
    attr_num = 4
    class_num = 3
    batch_10 = split_data_to_ten(iris_data)
    nbc = Normal_bayes_classifier()
    fbc = Flexiable_bayes_classfier()
    nbc_c_list = []
    fbc_c_list = []
    for i in range(10):
        temp = batch_10.copy()
        test_data = temp.pop(i)
        train_data = np.row_stack(temp)
        fbc.train_FBC(train_data.copy(), 3)
        nbc.train_NBC(train_data.copy(), 4,3)
        fbc_right = 0
        nbc_right = 0
        for test_item in test_data:
            label = test_item[-1]
            attr = test_item[:-1]
            fbc_pre = fbc.predict(attr)
            nbc_pre = nbc.predict_class(attr)
            if fbc_pre==label:
                fbc_right += 1
            if nbc_pre==label:
                nbc_right += 1
        fbc_correct = fbc_right/len(test_data)
        nbc_correct = nbc_right/len(test_data)
        print("pop:{} nbc:{}/{}={:.3f} fbc:{}/{}={:.3f}".format(i+1, nbc_right,
                                                        len(test_data), nbc_correct,
                                                        fbc_right,len(test_data), fbc_correct))
        nbc_c_list.append(nbc_correct)
        fbc_c_list.append(fbc_correct)
    print("avg= nbc:{:.3f} fbc:{:.3f}".format(sum(nbc_c_list)/len(nbc_c_list), sum(fbc_c_list)/len(fbc_c_list)))


    
# %%
