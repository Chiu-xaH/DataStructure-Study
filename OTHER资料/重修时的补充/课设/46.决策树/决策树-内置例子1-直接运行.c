#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#define TREE_IMAGE_PATH "D:\\tree.dot" // 生成可视化

typedef int Type;
#define N 1
#define Y 2
#define Nil 0
// 天气
typedef int Weather;
#define SUNNY 1
#define RAINY 2
#define CLOUDY 3
#define COUNT_FEATURE_WEATHER 3
// 温度
typedef int Temperature;
#define HOT 1
#define COLD 2
#define MEDIUM 3
#define COUNT_FEATURE_TEMP 3
// 湿度
typedef int Humidity;
#define HUMID 1
#define DRY 2
#define COUNT_FEATURE_HUMIDITY 2
// 风况
typedef int Wind;
#define WINDY 1
#define CALM 2
#define COUNT_FEATURE_WIND 2
// 特征索引
typedef int Feature;
#define FEATURE_WEATHER 0
#define FEATURE_TEMPERATURE 1
#define FEATURE_HUMIDITY 2
#define FEATURE_WIND 3
#define COUNT_FEATURE 4
// 训练数据集
typedef struct Data {
    Weather weather;
    Temperature temperature;
    Humidity humidity;
    Wind wind;
    Type classify;
} Data;
// Log2 数学公式
double log2_(double x) {
    return log(x) / log(2); 
}
// 计算熵 熵=求和(-p * logp)
double calcEntropy(Data* data, int size) {
    // Y的数量 N的数量
    int countY = 0, countN = 0;
    for (int i = 0; i < size; ++i) {
        if (data[i].classify == Y) countY++;
        else if (data[i].classify == N) countN++;
    }
    // 计算p（概率）=数量/总数量
    double py = (double)countY / size; // Y分类的p
    double pn = (double)countN / size; // N分类的p
    double entropy = 0; 
    if (py > 0) entropy -= py * log2_(py);
    if (pn > 0) entropy -= pn * log2_(pn);
    return entropy;
}
// 获取某个特征的最大可能取值
int getFeatureMaxValue(Feature featureIndex) {
    switch (featureIndex) {
        // 天气有三种
        case FEATURE_WEATHER: return COUNT_FEATURE_WEATHER;
        // ...
        case FEATURE_TEMPERATURE: return COUNT_FEATURE_TEMP;
        case FEATURE_HUMIDITY: return COUNT_FEATURE_HUMIDITY;
        case FEATURE_WIND: return COUNT_FEATURE_WIND;
    }
    return 0;
}
// 获取指定特征值对应的子数据集
int splitData(Data* data, int size, Feature featureIndex, Feature value, Data* out) {
    int count = 0;
    for (int i = 0; i < size; ++i) {
        int val;
        switch (featureIndex) {
            case FEATURE_WEATHER: val = data[i].weather; break;
            case FEATURE_TEMPERATURE: val = data[i].temperature; break;
            case FEATURE_HUMIDITY: val = data[i].humidity; break;
            case FEATURE_WIND: val = data[i].wind; break;
        }
        // 如果特征值匹配，则加入数组 
        // 例如featureIndex为FEATURE_WEATHER，将数据集的天气特征值赋给val，然后比较传入的特征值value是否匹配，是则加入数组
        if (val == value) {
            out[count++] = data[i];
        }
    }
    return count;
}
// 计算信息增益(IG) IG=父节点的熵 - 求和(子节点的熵 * 子结点的占比)
double calcGain(Data* data, int size, Feature featureIndex) {
    // 父结点的熵
    double entropyBefore = calcEntropy(data, size);
    // 子节点的熵
    double entropyAfter = 0.0;
    // 该Feature的最大有几种情况 例如天气Feature有晴、雨、多云，则为3
    // 此值将决定求和几次 或者说有几个子节点
    int maxValue = getFeatureMaxValue(featureIndex);
    // 子节点
    Data* subset = malloc(sizeof(Data) * size);
    for (int v = 1; v <= maxValue; v++) {
        int subsetSize = splitData(data, size, featureIndex, v, subset);
        // 子集为空，跳过
        if (subsetSize == 0) continue;
        // 子集的熵
        double subEntropy = calcEntropy(subset, subsetSize);
        // 求和 (对应上的数量/总数)*熵
        entropyAfter += ((double)subsetSize / size) * subEntropy;
    }
    free(subset);
    // IG
    return entropyBefore - entropyAfter;
}
// 判断是否为纯叶子 即所有数据都属于同一分类，无需再划分
bool isPure(Data* data, int size) {
    Type first = data[0].classify;
    for (int i = 1; i < size; ++i) {
        if (data[i].classify != first) return false;
    }
    return true;
}
// 决策树
typedef struct TreeNode {
    Feature featureIndex; // 按哪个特征划分
    Feature featureValue; // 当前节点是该特征的哪个取值（从父节点分裂过来）
    Type classify;    // 若是叶子，分类为N或Y
    int childrenCount;
    struct TreeNode** children; // 孩子数组，最大值应为特征的最大取值数getFeatureMaxValue(featureIndex)
    Feature* childFeatureValues; // 子节点对应的特征值
} TreeNode,*Tree;
// 建立决策树
TreeNode* buildTree(Data* data, int size, bool* usedFeatures) {
    // 新建结点
    TreeNode* node = malloc(sizeof(TreeNode));
    node->children = NULL;
    node->childFeatureValues = NULL;
    node->childrenCount = 0;
    node->featureValue = Nil;
    // 如果是纯净叶子结点，即已经为同一特征，无需再分，结束递归
    if (isPure(data, size)) {
        node->featureIndex = -1;
        node->classify = data[0].classify;
        return node;
    }
    // 寻找信息增益最大的特征及其信息增益值IG
    double bestGain = -1;
    Feature bestFeature = -1;
    for (int i = 0; i < COUNT_FEATURE; i++) {
        if (usedFeatures[i]) continue;
        double gain = calcGain(data, size, i);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeature = i;
        }
    }
    // 未找到最佳特征，说明无法再分裂，直接返回叶子结点
    if (bestFeature == -1) {
        node->featureIndex = -1;
        node->classify = data[0].classify;
        return node;
    }
    // 找到最佳特征，创建结点
    node->featureIndex = bestFeature;
    node->classify = Nil;
    // 标记该特征已使用
    usedFeatures[bestFeature] = true;
    // 该特征的最大取值数
    int maxVal = getFeatureMaxValue(bestFeature);
    // 分配maxVal个孩子结点 定义决策树时有说过
    node->children = malloc(sizeof(TreeNode*) * maxVal);
    // 分析到这里了
    node->childFeatureValues = malloc(sizeof(int) * maxVal);
    for (int v = 1; v <= maxVal; v++) {
        Data* subset = malloc(sizeof(Data) * size);
        int subsetSize = splitData(data, size, bestFeature, v, subset);
        if (subsetSize == 0) {
            node->children[v-1] = NULL;
            continue;
        }
        node->children[v-1] = buildTree(subset, subsetSize, usedFeatures);
        node->children[v-1]->featureValue = v;
        node->childFeatureValues[v-1] = v;
        node->childrenCount++;
        free(subset);
    }
    // 清理
    usedFeatures[bestFeature] = false;
    return node;
}
// 打印决策树
void printTree(TreeNode* node, int depth) {
    for (int i = 0; i < depth; i++) printf("  ");
    if (node->featureIndex == -1) {
        printf("Leaf: %s\n", node->classify == Y ? "Yes" : "No");
    } else {
        printf("Feature %d\n", node->featureIndex);
        int maxVal = getFeatureMaxValue(node->featureIndex);
        for (int i = 0; i < maxVal; i++) {
            if (node->children[i] == NULL) continue;
            for (int j = 0; j < depth + 1; j++) printf("  ");
            printf("-- Value %d -->\n", node->childFeatureValues[i]);
            printTree(node->children[i], depth + 2);
        }
    }
}
// 分类
Type classifyByTree(TreeNode* root, Data* sample) {
    TreeNode* node = root;
    while (node->featureIndex != -1) {
        int val;
        switch (node->featureIndex) {
            case FEATURE_WEATHER: val = sample->weather; break;
            case FEATURE_TEMPERATURE: val = sample->temperature; break;
            case FEATURE_HUMIDITY: val = sample->humidity; break;
            case FEATURE_WIND: val = sample->wind; break;
        }
        int maxVal = getFeatureMaxValue(node->featureIndex);
        if (val < 1 || val > maxVal || node->children[val - 1] == NULL) {
            return Nil; // 无法分类
        }
        node = node->children[val - 1];
    }
    return node->classify;
}

// 导出为.dot
void exportDot(TreeNode* node, FILE* fp, int* nodeId, int parentId) {
    int currentId = (*nodeId)++;

    if (node->featureIndex == -1) {
        fprintf(fp, "  node%d [label=\"Leaf: %s\", shape=box];\n", currentId, node->classify == Y ? "Yes" : "No");
    } else {
        fprintf(fp, "  node%d [label=\"Feature %d\"];\n", currentId, node->featureIndex);
    }

    if (parentId != -1) {
        fprintf(fp, "  node%d -> node%d;\n", parentId, currentId);
    }

    if (node->featureIndex != -1) {
        for (int i = 0; i < node->childrenCount; ++i) {
            if (node->children[i]) {
                fprintf(fp, "  node%d -> node%d [label=\"=%d\"];\n", currentId, *nodeId, node->childFeatureValues[i]);
                exportDot(node->children[i], fp, nodeId, currentId);
            }
        }
    }
}

void writeDotFile(TreeNode* root, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to create dot file");
        return;
    }

    fprintf(fp, "digraph DecisionTree {\n");
    int nodeId = 0;
    exportDot(root, fp, &nodeId, -1);
    fprintf(fp, "}\n");

    fclose(fp);
}

int main() {
    // 待分类数据集
    Data testData[] = {
        {SUNNY, HOT, DRY, CALM, Nil},
        {SUNNY, HOT, DRY, WINDY, Nil},
        {RAINY, HOT, DRY, CALM, Nil},
        {SUNNY, MEDIUM, DRY, CALM, Nil},
        {SUNNY, COLD, HUMID, WINDY, Nil},
        {SUNNY, COLD, HUMID, CALM, Nil}
    };
    // 训练数据集
    Data originalData[] = {
        {SUNNY, HOT, HUMID, CALM, N},
        {SUNNY, HOT, HUMID, WINDY, N},
        {CLOUDY, HOT, HUMID, CALM, Y},
        {RAINY, MEDIUM, HUMID, CALM, Y},
        {RAINY, COLD, DRY, CALM, Y},
        {RAINY, COLD, DRY, WINDY, N},
        {CLOUDY, COLD, DRY, WINDY, Y},
        {SUNNY, MEDIUM, HUMID, CALM, N},
        {SUNNY, COLD, DRY, CALM, Y},
        {RAINY, MEDIUM, DRY, CALM, Y},
        {SUNNY, MEDIUM, DRY, WINDY, Y},
        {CLOUDY, MEDIUM, HUMID, WINDY, Y},
        {CLOUDY, HOT, DRY, CALM, Y},
        {RAINY, MEDIUM, HUMID, WINDY, N},
    };
    int dataSize = sizeof(originalData) / sizeof(originalData[0]);
    // 用于标记特征是否已使用 索引即为特征
    bool usedFeatures[COUNT_FEATURE] = { false, false, false, false };
    // 建树
    TreeNode* root = buildTree(originalData, dataSize, usedFeatures);
    // 打印树
    // printTree(root, 0);
    // 分类
    printf("\nTest Data Classification:\n");
    int testSize = sizeof(testData) / sizeof(testData[0]);
    for (int i = 0; i < testSize; ++i) {
        Type result = classifyByTree(root, &testData[i]);
        printf("Test case %d => %s\n", i + 1, result == Y ? "Yes" : result == N ? "No" : "Unknown");
    }
    // 可视化图片
    writeDotFile(root, TREE_IMAGE_PATH);
    printf("Decision tree exported to %s\n",TREE_IMAGE_PATH);
    return 0;
}