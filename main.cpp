#pragma GCC optimize(3)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <limits>
#include <cassert>
#include <deque>
#include <random>

#define MAX_DISK_NUM 10
#define MAX_DISK_SIZE 16384
#define MAX_REQUEST_NUM 30000000
#define MAX_OBJECT_SIZE 5
#define MAX_OBJECT_NUM 100000
#define REP_NUM 3
#define FRE_PER_SLICING 1800
#define EXTRA_TIME 105

#define NONE -1

using namespace std;

// pair<int, int>的哈希函数
namespace std
{
    template <>
    struct hash<pair<int, int>>
    {
        size_t operator()(const pair<int, int>& p) const
        {
            return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
        }
    };
}

// 计算均值
double compute_mean(const vector<int>& l)
{
    double sum = 0.0;
    for (int v : l)
        sum += v;
    return sum / l.size();
}

// 计算两个等长列表的皮尔逊相关系数
double pearson_correlation(const vector<int>& row1, const vector<int>& row2)
{
    double mean1 = compute_mean(row1);
    double mean2 = compute_mean(row2);
    double numerator = 0.0, sum_den1 = 0.0, sum_den2 = 0.0;
    for (size_t i = 0; i < row1.size(); ++i)
    {
        numerator += (row1[i] - mean1) * (row2[i] - mean2);
        sum_den1 += pow(row1[i] - mean1, 2);
        sum_den2 += pow(row2[i] - mean2, 2);
    }
    double denominator = sqrt(sum_den1 * sum_den2);
    return denominator == 0 ? 1.0 : numerator / denominator;
}

// 计算皮尔逊相关系数矩阵
vector<vector<double>> pearson_matrix(const vector<vector<int>>& matrix)
{
    int num_rows = matrix.size();
    vector<vector<double>> correlations(num_rows, vector<double>(num_rows, 0.0));
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = i; j < num_rows; ++j)
        {
            if (i == j)
                correlations[i][j] = 1.0;
            else
            {
                double corr = (pearson_correlation(matrix[i], matrix[j]) + 1) * 0.5;
                correlations[i][j] = corr;
                correlations[j][i] = corr;
            }
        }
    }
    return correlations;
}

// 按照两个时间序列的峰值(最大值)位置计算相关系数矩阵，映射范围为[-1,1]
vector<vector<double>> peak_based_correlation_matrix(const vector<vector<int>>& matrix)
{
    int num_signals = matrix.size();

    // Special case: If there are 0 or 1 signals, return a 1x1 matrix
    if (num_signals <= 1)
    {
        return { {1} };
    }

    // Step 1: Extract the peak time indices (t_i) for each signal
    vector<int> peak_times(num_signals, 0);
    for (int i = 0; i < num_signals; ++i)
    {
        const auto& row = matrix[i];
        if (row.empty())
        {
            // If the signal is empty, set a special value, here we set 0
            peak_times[i] = 0;
        }
        else
        {
            // Find the index of the maximum value in the row as the peak position
            auto max_iter = max_element(row.begin(), row.end());
            peak_times[i] = distance(row.begin(), max_iter);
        }
    }

    // Step 2: Calculate the maximum reference time difference delta_T_max
    int min_t = *min_element(peak_times.begin(), peak_times.end());
    int max_t = *max_element(peak_times.begin(), peak_times.end());
    double delta_T_max = max_t - min_t;

    // If all the peaks are at the same time, return a matrix of all 1s
    if (delta_T_max == 0)
    {
        vector<vector<double>> correlations(num_signals, vector<double>(num_signals, 1.0));
        return correlations;
    }

    // Step 3: Compare peak time differences and map them to [-1, 1]
    vector<vector<double>> correlations(num_signals, vector<double>(num_signals, 0.0));

    for (int i = 0; i < num_signals; ++i)
    {
        for (int j = i; j < num_signals; ++j)
        {
            if (i == j)
            {
                // Same signal, diagonal should be 1
                correlations[i][j] = 1.0;
            }
            else
            {
                double dt = abs(peak_times[i] - peak_times[j]); // Time difference
                double corr = 0.0;

                if (dt >= delta_T_max)
                {
                    corr = -1.0;
                }
                else
                {
                    // Linear mapping: dt = 0 -> +1, dt = delta_T_max -> -1
                    corr = 1.0 - 2.0 * (dt / delta_T_max);
                }

                // Symmetric assignment
                correlations[i][j] = corr;
                correlations[j][i] = corr;
            }
        }
    }

    return correlations;
}

// 将磁盘平均分为k个页，将每个分页的起始和结束下标存储在一个vector中
vector<pair<int, int>> divide_into_segments(int n, int k)
{
    vector<pair<int, int>> segments;
    segments.reserve(k);

    int base_size = n / k;
    int start_idx = 0;

    for (int i = 0; i < k - 1; ++i)
    {
        int end_idx = start_idx + base_size;
        segments.emplace_back(start_idx, end_idx - 1);
        start_idx = end_idx;
    }
    segments.emplace_back(start_idx, n - 1);

    return segments;
}

// 一个时间周期内，从起始的token cost开始读动作，按照指数衰减的分数，计算可以一直读多久
int read_max_length(int remain_token, int last_read_token)
{
    int x = 0;
    while (remain_token >= last_read_token)
    {
        remain_token -= last_read_token;
        x += 1;
        last_read_token = max(static_cast<int>(ceil(last_read_token * 0.8)), 16);
    }

    return x;
}

// 存储单元的类，存放该单元的id，大小，标签，副本和存储位置
class Object
{
public:
    int object_id;
    int size;
    int tag;

    vector<int> replica;
    vector<vector<int>> unit;

    Object() {}

    Object(int object_id, int size, int tag)
        : object_id(object_id), size(size), tag(tag) {
    }

    void write(int disk, vector<int> position)
    {
        replica.emplace_back(disk);
        unit.emplace_back(position);
    }
};

// 标签的类，存放该标签的id，删除次数，写入次数和读取次数信息
class TagMessage
{
public:
    int tag;
    vector<int> delete_times;
    vector<int> write_times;
    vector<int> read_times;

    vector<pair<int, int>> space;         // tag所拥有的页

    TagMessage(int tag, vector<int> delete_times, vector<int> write_times, vector<int> read_times)
        : tag(tag), delete_times(delete_times), write_times(write_times), read_times(read_times) {
    }

    void add_space(int disk, int page)
    {
        space.emplace_back(disk, page);
    }

    const vector<pair<int, int>>& get_avail_pages() const
    {
        return space;
    }
};


class Disk
{
public:
    int name;                                       // 盘号
    int volume;                                     // 盘总容量
    int tokens;                                     // 每回合可以使用的token
    vector<pair<int, int>> data;                    // 盘中存储的数据
    unordered_map<pair<int, int>, int> index;       // 字典，根据数据索引位置
    int head[2];                                    // 磁头位置
    int read_length;                                // 能够读取的最远距离
    int last_read_token[2];                         // 记录上次读行动使用的token数，如果上次不是读行动，则为NONE

    int mask_count[2];                              // 每个磁头占有的mask数量
    int work_page[2];                               // 每个磁头的工作页

    int timestamp = 0;                              // 当前时间片
    int time_period = 0;                            // 当前周期， 以1800时间片为1周期
    vector<int> tasks_count;                        // 记录请求磁盘不同位置数据的请求量
    unordered_map<pair<int, int>, int>& task_mask;  // 引用自disk manager，所有数据的占用情况
    vector<int> index_mask;                         // 当前磁盘上每个下标的占用，用于应对有数据写入和删除的情况

    vector<int> page_load;                          // 每个页上需要读的物体总数
    vector<double> step_predict;                    // 每个页的预期完成时间

    vector<vector<int>> read_times;                 // 每个tag在每个周期的读取量

    int max_pages;                                  // 磁盘最大分页数量
    vector<int> pages_occupy;                       // 每个分页上存储的tag下标
    vector<pair<int, int>> pages;                   // 每个分页的起始和结束下标
    vector<int> pages_volume;                       // 每个页的剩余容量
    int base_volume;                                // 每个页的基本长度，用于计算下标所属的页

    int max_tag_num;                                // 最大tag数(legacy)
    vector<vector<double>> similarity;              // 不同tag读取次数的基于峰值的相似度，用于写过程中的分页选择(legacy)
    vector<vector<double>> similarity_for_disk;     // 不同tag读取次数的皮尔逊相似度，用于写过程中的分盘选择
    double base_ability;                            // 计算磁盘的基本读取能力

    vector<int> read_plan;                          // 根据当前存储的tag，估计每个周期的读取量(legacy)
    vector<vector<vector<double>>> similarity_decay;// 随距离衰减的相似度
    vector<vector<double>> position_value;          // 每个位置相似度的总和(legacy)

    // for tokens = 340
    vector<double> readability = { 340.0, 97.772, 74.811, 54.526, 44.344, 38.472, 34.279, 31.463, 28.77, 27.675, 26.404, 25.994, 24.818, 24.079, 23.588, 23.05, 22.974, 22.348, 22.353, 21.962, 21.79, 21.813, 21.595, 21.526, 21.509, 21.398, 21.334, 21.149, 21.201, 21.162, 21.14, 21.116, 21.1, 21.113, 21.059, 21.042, 21.046, 21.04, 21.038, 21.015, 21.02, 21.023, 21.008, 21.006, 20.998, 21.007, 20.996, 20.998, 21.002, 21.0, 20.998, 20.996, 20.996, 20.996, 20.997, 20.996, 20.996, 20.996, 20.996, 20.997, 20.996, 20.997, 20.996, 20.996, 20.996, 20.996, 20.996, 20.997, 20.997, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.997, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996, 20.996 };


    Disk(int name, int volume, int tokens, int page_num,
        const vector<vector<double>>& similarity_for_disk,
        const vector<vector<double>>& similarity,
        const vector<vector<vector<double>>>& similarity_decay,
        const vector<vector<int>>& read_times,
        unordered_map<pair<int, int>, int>& task_mask)
        : name(name), volume(volume), tokens(tokens),
        similarity_for_disk(similarity_for_disk),
        similarity(similarity), similarity_decay(similarity_decay),
        read_times(read_times), task_mask(task_mask)

    {
        // 初始化所有量
        head[0] = 0;
        head[1] = 0;
        last_read_token[0] = NONE;
        last_read_token[1] = NONE;

        mask_count[0] = 0;
        mask_count[1] = 0;
        work_page[0] = NONE;
        work_page[1] = NONE;
        index_mask.resize(volume, 0);

        data.resize(volume, { NONE, NONE });
        read_length = read_max_length(tokens, 64);

        tasks_count.resize(volume, 0);
        page_load.resize(page_num, 0);
        step_predict.resize(page_num, 0);

        max_pages = page_num;
        pages_occupy.resize(max_pages, NONE);
        pages = divide_into_segments(volume, max_pages);
        for (const auto& page : pages)
        {
            int start = page.first;
            int end = page.second;
            pages_volume.emplace_back(end - start + 1);
        }
        base_volume = int(volume / max_pages);

        max_tag_num = similarity.size();
        base_ability = 1800.0 * tokens / 64.0;

        read_plan.assign(read_times[0].size(), 0);
        position_value.resize(max_tag_num, vector<double>(max_pages, 1.0));
    }

    // 指派任务
    void assign_task(const pair<int, int>& task)
    {
        int task_index = index[task];
        // tasks_queue[104][task_index]++;
        tasks_count[task_index] += 1;

        if (tasks_count[task_index] == 1) {
            page_load[min(task_index / base_volume, max_pages - 1)] += 1;
        }

    }

    // 取消任务
    void cancel_task(const pair<int, int>& task)
    {
        int task_index = index[task];
        // completed_tasks[task_index] += tasks_count[task_index];
        if (tasks_count[task_index] != 0) {
            page_load[min(task_index / base_volume, max_pages - 1)] -= 1;
        }
        tasks_count[task_index] = 0;
    }

    // 取消特定数量的任务
    void cancel_expired_task(const pair<int, int>& task, int cancel_num)
    {
        int task_index = index[task];
        // completed_tasks[task_index] += tasks_count[task_index];
        tasks_count[task_index] -= cancel_num;
        if (tasks_count[task_index] == 0) {
            page_load[min(task_index / base_volume, max_pages - 1)] -= 1;
        }
    }

    // 读策略
    pair<vector<pair<int, int>>, int> simple_read(int head_index, int remain_token) {
        vector<pair<int, int>> complete_tasks;

        for (int index = 0;; ++index)
        {
            if (tasks_count[head[head_index]])
            {
                int token_cost = 64;
                if (last_read_token[head_index] != NONE)
                {
                    token_cost = max(static_cast<int>(ceil(last_read_token[head_index] * 0.8)), 16);
                }

                if (token_cost <= remain_token)
                {
                    printf("r");
                    remain_token -= token_cost;
                }
                else
                    break;

                last_read_token[head_index] = token_cost;
                complete_tasks.emplace_back(data[head[head_index]]);

                // 立即处理已经完成的任务，清除tasks count
                if (tasks_count[head[head_index]] != 0) {
                    page_load[min(head[head_index] / base_volume, max_pages - 1)] -= 1;
                }
                tasks_count[head[head_index]] = 0;
            }
            else
            {
                // 如果不需要读，则根据后续的价值判断当前是否要延续读
                if (last_read_token[head_index] == NONE)
                {
                    if (remain_token >= 1)
                    {
                        printf("p");
                        remain_token -= 1;
                    }
                    else
                        break;
                }
                else
                {
                    // 计算剩余的token数在当前的读cost下还能读多少步
                    int continue_read_len = read_max_length(remain_token + tokens, last_read_token[head_index]);
                    int continue_read_obj = 0;
                    for (int i = head[head_index]; i < head[head_index] + continue_read_len; ++i)
                    {
                        continue_read_obj += int(bool(tasks_count[i % volume]));
                    }

                    int pass_obj = 0;
                    for (int i = 1; i <= remain_token + tokens - 64; ++i)
                    {
                        if (tasks_count[(head[head_index] + i) % volume])
                        {
                            int pass_len = read_max_length(remain_token + tokens - i, 64);
                            for (int j = head[head_index] + i; j <= head[head_index] + i + pass_len; ++j)
                            {
                                pass_obj += int(bool(tasks_count[j % volume]));
                            }
                            break;
                        }
                    }

                    if (continue_read_obj >= pass_obj && continue_read_obj > 0)
                    {
                        int token_cost = max(static_cast<int>(ceil(last_read_token[head_index] * 0.8)), 16);
                        if (token_cost <= remain_token)
                        {
                            printf("r");
                            remain_token -= token_cost;
                            last_read_token[head_index] = token_cost;

                            if (tasks_count[head[head_index]]) {
                                complete_tasks.emplace_back(data[head[head_index]]);
                                page_load[min(head[head_index] / base_volume, max_pages - 1)] -= 1;
                                tasks_count[head[head_index]] = 0;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {

                        last_read_token[head_index] = NONE;
                        int token_cost = 1;
                        if (token_cost <= remain_token)
                        {
                            printf("p");
                            remain_token -= token_cost;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }

            if (data[head[head_index]].first != NONE) {
                task_mask[data[head[head_index]]] -= 1;
            }

            mask_count[head_index] -= 1;
            index_mask[head[head_index]] -= 1;
            head[head_index] = (head[head_index] + 1) % volume;

            if (mask_count[head_index] == 0) {
                work_page[head_index] = NONE;
                break;
            }
        }

        return { complete_tasks, remain_token };
    }

    // 执行任务
    vector<pair<int, int>> work()
    {
        vector<pair<int, int>> complete_tasks;

        // 搜索的最大距离
        int max_range = min(tokens - 64, volume - 1);

        for (int head_index = 0; head_index < 2; head_index++) {
            int remain_token = tokens;

            while (true) {
                // 如果没有工作，就分配一个工作
                if (work_page[head_index] == NONE) {
                    get_mask(head_index);
                }

                // 如果分配了工作，但是不在对应的页内，就想办法到达页的起始位置
                int page_index = work_page[head_index];
                if (min(head[head_index] / base_volume, max_pages - 1) != page_index) {
                    last_read_token[head_index] = NONE;

                    int distance = (pages[page_index].first - head[head_index] + volume) % volume;

                    if (distance <= remain_token) {
                        // 如果能直接p过去，就p过去
                        for (int i = 0; i < distance; i++) {
                            printf("p");
                        }
                        head[head_index] = (head[head_index] + distance) % volume;
                        remain_token -= distance;
                    }
                    else if (remain_token == tokens) {
                        // 如果不能p过去，就尝试跳跃过去
                        printf("j %d\n", pages[page_index].first + 1);
                        head[head_index] = pages[page_index].first;
                        remain_token = 0;
                        break;
                    }
                    else {
                        // 如果也不能跳过去，就尽可能pass
                        for (int i = 0; i < remain_token; i++) {
                            printf("p");
                        }
                        head[head_index] = (head[head_index] + remain_token) % volume;
                        remain_token = 0;
                    }
                }

                if (remain_token == 0) {
                    printf("#\n");
                    break;
                }

                pair<vector<pair<int, int>>, int> read_result = simple_read(head_index, remain_token);
                vector<pair<int, int>> tmp_complete_tasks = read_result.first;
                complete_tasks.insert(complete_tasks.end(), tmp_complete_tasks.begin(), tmp_complete_tasks.end());
                remain_token = read_result.second;

                if (remain_token == 0 || mask_count[head_index] != 0) {
                    printf("#\n");
                    break;
                }
            }
        }

        return complete_tasks;
    }

    // 根据价值选择一个最好的页，优先选择没有被占据的页
    pair<int, int> select_page(int tag)
    {
        double max_value = -INFINITY;
        int max_value_index = NONE;

        double max_replace_value = -INFINITY;
        int max_replace_index = NONE;

        for (int i = 0; i < max_pages; ++i)
        {
            double value;
            if (double(pages_volume[i]) / base_volume < 0.05)
            {
                value = -INFINITY;
            }
            else
            {
                value = (position_value[tag][i]) * pages_volume[i];
            }

            if (pages_occupy[i] == NONE && value > max_value)
            {
                max_value = value;
                max_value_index = i;
            }
            if (value > max_replace_value)
            {
                max_replace_value = value;
                max_replace_index = i;
            }
        }

        if (max_value_index != NONE)
        {
            return { max_value_index, NONE };
        }
        else
        {
            return { max_replace_index, pages_occupy[max_replace_index] };
        }
    }

    // 申请页，并修改价值矩阵
    pair<int, int> malloc_space(int tag)
    {
        // // 反转了申请逻辑，改为让相似的标签分开
        pair<int, int> result = select_page(tag);
        int page_index = result.first;
        int replace_tag = result.second;
        if (pages_occupy[page_index] != NONE)
        {
            int old_tag = pages_occupy[page_index];
            for (int i = 0; i < max_tag_num; ++i)
                for (int j = 0; j < max_pages; ++j)
                {
                    int d = min((page_index - j + max_pages) % max_pages,
                        (j - page_index + max_pages) % max_pages);
                    position_value[i][j] -= similarity_decay[d][i][old_tag];
                }

            for (int i = 0; i < read_plan.size(); ++i)
                read_plan[i] -= read_times[old_tag][i];
        }

        pages_occupy[page_index] = tag;

        for (int i = 0; i < max_tag_num; ++i)
            for (int j = 0; j < max_pages; ++j)
            {
                int d = min((page_index - j + max_pages) % max_pages,
                    (j - page_index + max_pages) % max_pages);
                position_value[i][j] += similarity_decay[d][i][tag];
            }

        for (int i = 0; i < read_plan.size(); ++i)
            read_plan[i] += read_times[tag][i];

        return { page_index, replace_tag };
    }

    // 只申请空间，不写入数据
    vector<int> find_write_space(int size, int page_id) {
        int start = pages[page_id].first;
        int end = pages[page_id].second;
        vector<int> unit;
        int current_size = 0;

        for (int i = start; i <= end; ++i)
        {
            if (data[i].first == NONE)
            {
                unit.emplace_back(i);
                if (++current_size == size)
                    break;
            }
        }

        if (current_size < size)
            return vector<int>{NONE};
        else {
            return unit;
        }
    }

    // 直接写入unit
    void write_unit(vector<int> unit, int object_id, int page_id) {
        for (int i = 0; i < unit.size(); ++i)
        {
            pages_volume[page_id]--;
            data[unit[i]] = make_pair(object_id, i);
            index[{object_id, i}] = unit[i];

            if (index_mask[unit[i]] != 0) {
                task_mask[data[unit[i]]] += index_mask[unit[i]];
            }
        }
    }

    // 将数据写入特定的页中
    vector<int> write_in_page(int object_id, int size, int page_id)
    {
        vector<int> unit = find_write_space(size, page_id);
        if (unit[0] == NONE) {
            return unit;
        }

        for (int i = 0; i < size; ++i)
        {
            pages_volume[page_id]--;
            data[unit[i]] = make_pair(object_id, i);
            index[{object_id, i}] = unit[i];

            if (index_mask[unit[i]] != 0) {
                task_mask[data[unit[i]]] += index_mask[unit[i]];
            }
        }

        return unit;
    }

    // 删除特定unit的数据
    void free_space(const vector<int>& unit)
    {
        for (int idx : unit)
        {
            int page = min(idx / base_volume, max_pages - 1);
            pages_volume[page]++;
            if (data[idx] != make_pair(NONE, NONE))
            {
                index.erase(data[idx]);
            }

            task_mask[data[idx]] = 0;
            data[idx] = make_pair(NONE, NONE);

        }
    }

    // 转移数据，转移时，只给出目标页，具体位置由磁盘自主计算
    vector<int> transfer_to_page(int object_id, int size, int page_id, const vector<int>& ori_unit, int ori_page_id)
    {
        int start = pages[page_id].first;
        int end = pages[page_id].second;
        vector<int> unit;
        int current_size = 0;

        for (int i = start; i <= end; ++i)
        {
            if (data[i].first == NONE)
            {
                unit.emplace_back(i);
                if (++current_size == size)
                    break;
            }
        }

        if (current_size < size)
            return vector<int>{NONE};

        for (int i = 0; i < size; ++i)
        {
            pages_volume[page_id]--;
            pages_volume[ori_page_id]++;

            data[unit[i]] = make_pair(object_id, i);
            data[ori_unit[i]] = make_pair(NONE, NONE);

            index[{object_id, i}] = unit[i];

            if (tasks_count[ori_unit[i]] > 0) {
                tasks_count[unit[i]] = tasks_count[ori_unit[i]];
                tasks_count[ori_unit[i]] = 0;

                page_load[page_id]++;
                page_load[ori_page_id]--;
            }
            task_mask[{object_id, i}] += index_mask[unit[i]] - index_mask[ori_unit[i]];
        }
        return unit;
    }

    // 直接写入到指定位置
    void transfer_to_unit(vector<int> unit, int object_id, const vector<int>& ori_unit)
    {
        int size = unit.size();
        for (int i = 0; i < size; ++i)
        {
            data[unit[i]] = make_pair(object_id, i);
            data[ori_unit[i]] = make_pair(NONE, NONE);

            index[{object_id, i}] = unit[i];

            if (tasks_count[ori_unit[i]] > 0) {
                tasks_count[unit[i]] = tasks_count[ori_unit[i]];
                tasks_count[ori_unit[i]] = 0;
            }
            task_mask[{object_id, i}] += index_mask[unit[i]] - index_mask[ori_unit[i]];
        }
    }

    int page_volume(int page)
    {
        return pages_volume[page];
    }

    // 该磁盘的优势值，用于磁盘选择
    double advantage(int tag)
    {
        double rest_volume = 0;
        for (int v : pages_volume)
            rest_volume += v;

        if (rest_volume / volume <= 0.01)
        {
            return -1000000;
        }

        double tmp_similarity = 0;
        for (auto tag_curr : pages_occupy)
        {
            if (tag_curr == NONE)
            {
                continue;
            }
            tmp_similarity -= similarity_for_disk[tag][tag_curr];
        }

        return tmp_similarity;
    }

    // 计算一个坐标位于哪个页，以及在页内的偏移量
    pair<int, int> index_position(int index) {
        int page_index = min(index / base_volume, max_pages - 1);
        int offset = index - page_index * base_volume;

        return { page_index, offset };
    }

    // 估算在当前状况下，某个下标的任务需要等待的时间
    double estimate_steps(int index) {
        int head_index = ((index - head[0]) % volume < (index - head[1]) % volume) ? 0 : 1;

        pair<int, int> head_position = index_position(head[head_index]);
        int head_page_index = head_position.first;
        int head_offset = head_position.second;

        pair<int, int> position = index_position(index);
        int index_page_index = position.first;
        int index_offset = position.second;

        if (head_page_index == index_page_index && index_offset >= head_offset) {
            return double(index_offset - head_offset) / readability[max(static_cast<int>(round(100.0 * page_load[head_page_index] / base_volume)), 100)];
        }

        double steps = 0.0;
        steps += double(base_volume - head_offset) / readability[max(static_cast<int>(round(100.0 * page_load[head_page_index] / base_volume)), 100)];
        steps += double(index_offset) / readability[max(static_cast<int>(round(100.0 * page_load[index_page_index] / base_volume)), 100)];

        for (int page_index = (head_page_index + 1) % max_pages; page_index != index_page_index; page_index = (page_index + 1) % max_pages) {
            steps += double(base_volume) / readability[max(static_cast<int>(round(100.0 * page_load[page_index] / base_volume)), 100)];
        }

        return steps;
    }

    int get_data_index(pair<int, int> data) {
        return index[data];
    }

    // 计算一个页中没有被占有的数据价值
    double masked_page_value(int page_index) {
        int page_start = pages[page_index].first;
        int page_end = pages[page_index].second;

        double value = 0.0;
        for (int index = page_start; index <= page_end; index++) {
            if (data[index].first == NONE) {
                continue;
            }

            if (task_mask[data[index]] == 0) {
                value += tasks_count[index];
            }
        }

        return value;
    }

    // 根据页价值选择页
    int best_masked_page(int head_index) {
        int head_page = min(head[head_index] / base_volume, max_pages - 1);

        double best_value = NONE;
        int best_index = NONE;

        for (int page_index = 0; page_index < max_pages; page_index++) {
            double page_value = masked_page_value(page_index);
            if (page_index == head_page) {
                // 邻近的页给个buff
                page_value *= 1.2;
            }

            if (page_value > best_value) {
                best_value = page_value;
                best_index = page_index;
            }
        }
        return best_index;
    }

    // 为磁头选择页，并获取mask，同步到disk manager
    void get_mask(int head_index) {
        int page_index = best_masked_page(head_index);
        work_page[head_index] = page_index;

        int page_start = pages[page_index].first;
        int page_end = pages[page_index].second;

        for (int index = page_start; index <= page_end; index++) {

            mask_count[head_index] += 1;
            index_mask[index] += 1;
            if (data[index].first != NONE) {
                task_mask[data[index]] += 1;
            }
        }
    }

    // 更新每个页之前的预期步数
    void update_step_predict() {
        vector<pair<int, double>> page_value;
        for (int page_index = 0; page_index < max_pages; page_index++) {
            double value = masked_page_value(page_index);
            page_value.emplace_back(make_pair(page_index, value));
        }

        std::stable_sort(page_value.begin(), page_value.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second; // 按值（second）降序排序
            });

        double current_load = 0;
        for (int head_index = 0; head_index < 2; head_index++) {
            if (work_page[head_index] == NONE) {
                continue;
            }
            int head_offset = head[head_index] - base_volume * work_page[head_index];
            current_load += double(base_volume - head_offset) / readability[max(static_cast<int>(round(100.0 * page_load[work_page[head_index]] / base_volume)), 100)];
        }

        current_load /= 2;

        for (auto item : page_value) {
            int page_index = item.first;

            step_predict[page_index] = current_load;

            double current_page_volume = double(pages[page_index].second - pages[page_index].first + 1);
            double step = current_page_volume / readability[round(100.0 * page_load[page_index] / current_page_volume)];

            current_load += step / 2;
        }
    }

    // 估计一个任务的完成步数
    double predict_complete_time(pair<int, int> task) {
        int data_index = index[task];
        int page_index = min(data_index / base_volume, max_pages - 1);

        double current_page_volume = double(pages[page_index].second - pages[page_index].first + 1);
        double step = 0;

        if (index_mask[data_index]) {
            int distance = min({ data_index - pages[page_index].first, (data_index - head[0] + volume) % volume, (data_index - head[1] + volume) % volume });
            step += distance / readability[round(100.0 * page_load[page_index] / current_page_volume)];
        }
        else {
            step += step_predict[page_index];
            step += (data_index - pages[page_index].first) / readability[round(100.0 * page_load[page_index] / current_page_volume)];
        }

        return step;
    }

};


// 定义请求节点
struct RequestNode {
    int id;              // 请求 ID
    int timestamp;       // 时间戳
    int obj_id;          // 对象 ID
    RequestNode* prev;   // 前一个请求
    RequestNode* next;   // 后一个请求

    // 构造函数
    RequestNode(int _id, int _timestamp, int _obj_id)
        : id(_id), timestamp(_timestamp), obj_id(_obj_id), prev(nullptr), next(nullptr) {
    }
};

// 定义请求管理器
class RequestManager {
public:
    RequestNode* head;                         // 链表头哨兵节点
    RequestNode* tail;                         // 链表尾哨兵节点
    unordered_map<int, RequestNode*> map; // 哈希表，用于通过 ID 快速查找请求

    // 构造函数
    RequestManager() {
        head = new RequestNode(NONE, NONE, NONE); // 头哨兵节点，ID、timestamp 和 obj_id 为无效值
        tail = new RequestNode(NONE, NONE, NONE); // 尾哨兵节点，ID、timestamp 和 obj_id 为无效值
        head->next = tail;
        tail->prev = head;
    }

    // 析构函数，释放所有动态分配的内存
    ~RequestManager() {
        RequestNode* current = head;
        while (current) {
            RequestNode* nextNode = current->next;
            delete current;
            current = nextNode;
        }
    }

    // 插入请求
    void addRequest(int id, int timestamp, int obj_id) {
        // 创建新节点
        RequestNode* newNode = new RequestNode(id, timestamp, obj_id);

        // 插入到链表末尾（尾哨兵节点之前）
        newNode->prev = tail->prev;
        newNode->next = tail;
        tail->prev->next = newNode;
        tail->prev = newNode;

        // 更新哈希表
        map[id] = newNode;
    }

    // 删除请求
    void removeRequest(int req_id) {

        // 获取请求节点
        RequestNode* node = map[req_id];

        // 从链表中移除节点
        node->prev->next = node->next;
        node->next->prev = node->prev;

        // 从哈希表中移除
        map.erase(req_id);

        // 释放节点内存
        delete node;
    }

    // 获取过期请求统计并删除过期请求
    unordered_map<int, int> ExpiredRequests(int current_timestamp) {
        unordered_map<int, int> expired_count; // 存储 obj_id 的计数
        int max_time = 104; // 大于104时间片没有完成的，直接上报繁忙

        // 从链表头部开始遍历
        RequestNode* current = head->next;
        while (current != tail) {
            if (current_timestamp - current->timestamp > max_time) {
                // 统计 obj_id
                expired_count[current->obj_id]++;

                // 记录下一个节点
                RequestNode* next_node = current->next;

                // 从链表中移除当前节点
                current->prev->next = current->next;
                current->next->prev = current->prev;

                // 从哈希表中移除
                map.erase(current->id);

                // 释放当前节点内存
                delete current;

                // 移动到下一个节点
                current = next_node;
            }
            else {
                // 一旦遇到未过期的节点，停止遍历
                break;
            }
        }

        return expired_count;
    }

};



class DiskManager
{
public:
    int PAGES_NUM = 20;             // 每个磁盘的页数，最好值介于20-24之间
    int timestamp;                  // 当前时间片
    int time_num;                   // 总时间片数量
    int tag_num;                    // 总tag数量
    int disk_num;                   // 总磁盘数量
    int disk_volume;                // 磁盘的容量
    int disk_tokens;                // 磁盘可以使用的token数
    int exchange_num;               // 每次垃圾回收可以交换的次数
    vector<int> disk_tokens_list;   // 每个磁盘的token数

    unordered_map<pair<int, int>, int> task_mask;       // 数据的占有情况

    vector<TagMessage> tags;        // 每个tag的信息

    int rep_num = 3;                // 副本数量
    int fre_per_slicing = 1800;     // 每个周期的时间片数

    vector<Disk> disks;                                                 // 磁盘对象
    unordered_map<int, Object> objects;                                 // 文件对象
    unordered_map<int, deque<pair<int, set<int>>>> req_object_ids;      // 需要读取某个文件的所有请求以及实时完成情况
    vector<vector<vector<int>>> full_req;                               // 所有请求（以块为单位）和指派的磁盘

    RequestManager request_manager;                                     // 一种高效（？）的请求管理方法，结构为哈希双向链表

    vector<int> busy_req;                                               // 繁忙的请求

    DiskManager(int T, int M, int N, int V, int G, int K, const vector<TagMessage>& tag_list, vector<int> g)
        : time_num(T), tag_num(M), disk_num(N), disk_volume(V), disk_tokens(G), exchange_num(K), tags(tag_list), disk_tokens_list(g)
    {
        timestamp = 0;
        preprocess();

        printf("OK\n");
        fflush(stdout);
    }

    void preprocess()
    {
        vector<vector<int>> read_times;
        vector<vector<int>> write_times;
        vector<vector<int>> delete_times;
        for (auto& tag : tags)
        {
            read_times.emplace_back(tag.read_times);
            write_times.emplace_back(tag.write_times);
            delete_times.emplace_back(tag.delete_times);
        }

        vector<int> all_keep_blocks;
        all_keep_blocks.resize(write_times[0].size(), 0);
        for (int index = 0; index < write_times[0].size(); index++)
        {
            for (int tag = 0; tag < write_times.size(); tag++)
            {
                all_keep_blocks[index] += write_times[tag][index] - delete_times[tag][index];
            }

            if (index != 0)
            {
                all_keep_blocks[index] += all_keep_blocks[index - 1];
            }
        }

        // 计算读取阈值
        vector<int> all_read_times;
        all_read_times.resize(read_times[0].size(), 0);
        for (int index = 0; index < read_times[0].size(); index++)
        {
            for (auto& tag_read : read_times)
            {
                all_read_times[index] += tag_read[index];
            }
        }

        // 计算相似矩阵
        vector<vector<double>> similarity_for_disk = pearson_matrix(read_times);
        vector<vector<double>> similarity = peak_based_correlation_matrix(read_times);

        vector<vector<vector<double>>> similarity_decay;
        int tag_num = read_times.size();
        int max_pages = PAGES_NUM;
        int furthest = max_pages / 2;

        for (int index = 0; index <= furthest; ++index)
        {
            vector<vector<double>> tmp(tag_num, vector<double>(tag_num, 0));
            double penalty = 0.5 * exp(-0.5 * index);
            /*double penalty = 0.0;*/
            for (int i = 0; i < tag_num; ++i)
                for (int j = 0; j < tag_num; ++j)
                    tmp[i][j] = similarity[i][j] * (furthest - index) / furthest - penalty;
            similarity_decay.emplace_back(tmp);
        }

        // 初始化磁盘对象
        for (int i = 0; i < disk_num; ++i)
        {
            disks.emplace_back(Disk(i, disk_volume, disk_tokens,
                max_pages, similarity_for_disk, similarity, similarity_decay,
                read_times, task_mask));
        }

        // 申请full_req空间
        full_req.resize(MAX_OBJECT_NUM);
        for (int i = 0; i < MAX_OBJECT_NUM; i++)
        {
            full_req[i].resize(MAX_OBJECT_SIZE);
        }
        for (int i = 0; i < MAX_OBJECT_NUM; ++i)
        {
            for (int j = 0; j < MAX_OBJECT_SIZE; ++j)
            {
                // full_req[i][j] = NONE;
                vector<int> tmp;
                full_req[i][j] = tmp;
            }
        }
    }

    // 处理过期请求
    void process_expired_object(int expired_object_id, int expired_count) {
        // 结果存储
        unordered_map<int, int> element_count;

        // 获取对应 deque 的引用
        deque<pair<int, set<int>>>& dq = req_object_ids[expired_object_id];

        // 遍历前 k 项
        int count = 0;
        for (auto it = dq.begin(); it != dq.end() && count < expired_count; ++it, ++count) {
            const auto& item = *it;

            busy_req.push_back(item.first);

            // 统计 set<int> 中的元素
            for (int elem : item.second) {
                element_count[elem]++;
            }
        }

        // 删除前 k 项
        dq.erase(dq.begin(), dq.begin() + expired_count);

        int tag = objects[expired_object_id].tag;

        // 处理已经发出的请求
        for (int disk_index : objects[expired_object_id].replica) {
            for (auto item = element_count.begin(); item != element_count.end(); item++) {
                disks[disk_index].cancel_expired_task(make_pair(expired_object_id, item->first), item->second);
            }
        }

    }

    void timestamp_action(int ts)
    {

        int QUEUE_MAX_SIZE = 105;
        timestamp = ts;
        printf("TIMESTAMP %d\n", ts);

        // 更新每个盘上的时间信息
        for (auto& disk : disks)
        {
            disk.timestamp = ts;
            if (ts % 1800 == 0)
            {
                disk.time_period += 1;
            }
            if (ts % 1800 == 1)
            {
                disk.tokens = disk_tokens + disk_tokens_list[disk.time_period];
            }
        }

        fflush(stdout);
    }

    void delete_action(const vector<int>& delete_ids)
    {
        vector<int> abort_req;
        abort_req.reserve(1000);

        for (int delete_id : delete_ids)
        {
            int object_size = objects[delete_id].size;
            // 取消每个块在每个盘上的任务
            for (int block = 0; block < object_size; ++block)
            {
                pair<int, int> task = { delete_id, block };

                for (auto disk_num : full_req[delete_id][block])
                {
                    disks[disk_num].cancel_task(task);
                }

                full_req[delete_id][block].clear();
            }

            // 计算被取消的请求
            for (auto& req : req_object_ids[delete_id])
            {
                abort_req.emplace_back(get<0>(req));
                request_manager.removeRequest(get<0>(req));
            }
            req_object_ids.erase(delete_id);

            // 删除数据
            for (size_t i = 0; i < objects[delete_id].replica.size(); ++i)
            {
                int disk_index = objects[delete_id].replica[i];
                const vector<int>& unit = objects[delete_id].unit[i];
                disks[disk_index].free_space(unit);
            }

            objects.erase(delete_id);
        }

        printf("%zu\n", abort_req.size());
        for (int req : abort_req) {
            printf("%d\n", req);
        }
        fflush(stdout);
    }

    // 申请空间，根据优势选择磁盘和页
    pair<int, int> select_disk(int tag, const set<int>& exclude)
    {
        double best_advantage = -numeric_limits<double>::infinity();
        int best_index = NONE;

        for (int i = 0; i < disks.size(); ++i)
        {
            if (exclude.count(i))
                continue;
            double adv = disks[i].advantage(tag);
            if (adv > best_advantage)
            {
                best_advantage = adv;
                best_index = i;
            }
        }

        pair<int, int> result = disks[best_index].malloc_space(tag);
        int page_index = result.first;
        int replace_tag = result.second;

        if (replace_tag != NONE)
        {
            auto it = find(tags[replace_tag].space.begin(), tags[replace_tag].space.end(), make_pair(best_index, page_index));
            if (it != tags[replace_tag].space.end())
            {
                tags[replace_tag].space.erase(it);
            }
        }

        return { best_index, page_index };
    }

    // 妙妙函数，本意是尽可能写满一个页
    double value_function(int pagevolume)
    {
        double max_page_volume = double(disk_volume) / tag_num / 2;
        double divide = 0.85 * max_page_volume;
        return -abs(pagevolume - divide);

        if (pagevolume < divide)
        {
            return (max_page_volume - pagevolume) / (max_page_volume - divide);
        }
        else
        {
            return pagevolume / divide;
        }
    }

    void write_action(const vector<tuple<int, int, int>>& write_data)
    {
        for (const auto& item : write_data)
        {
            int object_id = get<0>(item);
            int size = get<1>(item);
            int tag = get<2>(item);

            printf("%d\n", object_id);
            objects[object_id] = Object(object_id, size, tag);
            TagMessage* tag_item = &tags[tag];
            auto avail_pages = tag_item->get_avail_pages();

            // 筛选所有页
            stable_sort(avail_pages.begin(), avail_pages.end(),
                [this](const pair<int, int>& a, const pair<int, int>& b)
                {
                    return disks[a.first].page_volume(a.second) < disks[b.first].page_volume(b.second);
                });

            int copy_num = 0;
            set<int> write_disks;
            vector<pair<int, vector<int>>> write_history;

            // 尝试写入
            for (const auto& item : avail_pages)
            {
                int disk_index = item.first;
                int page_index = item.second;

                if (write_disks.count(disk_index))
                    continue;

                auto result = disks[disk_index].write_in_page(object_id, size, page_index);
                if (result[0] == NONE)
                    continue;

                copy_num++;
                write_disks.insert(disk_index);
                objects[object_id].write(disk_index, result);
                write_history.emplace_back(disk_index, result);
                if (copy_num == 3)
                    break;
            }

            // 如果还有副本没有写入，则申请新的页
            while (copy_num < 3)
            {
                pair<int, int> select_result = select_disk(tag, write_disks);


                int disk_index = select_result.first;
                int page_index = select_result.second;

                auto result = disks[disk_index].write_in_page(object_id, size, page_index);
                assert(result[0] != NONE);

                tag_item->add_space(disk_index, page_index);
                copy_num++;
                write_disks.insert(disk_index);
                objects[object_id].write(disk_index, result);
                write_history.emplace_back(disk_index, result);
            }

            for (const auto& item : write_history)
            {
                int disk_index = item.first;
                vector<int> unit = item.second;

                printf("%d ", disk_index + 1);
                for (int u : unit) {
                    printf("%d ", u + 1);
                }
                printf("\n");
            }
        }

        fflush(stdout);
    }

    void assign_tasks(const vector<pair<int, int>>& read_datas)
    {
        for (const auto& item : read_datas)
        {
            int request_id = item.first;
            int object_id = item.second;
            int object_size = objects[object_id].size;
            int tag = objects[object_id].tag;


            request_manager.addRequest(request_id, timestamp, object_id);


            // 登记请求和请求完成情况
            pair<int, set<int>> req_info;
            req_info.first = request_id;
            set<int> block_set;
            for (int i = 0; i < object_size; ++i)
                block_set.insert(i);
            req_info.second = move(block_set);
            req_object_ids[object_id].emplace_back(move(req_info));

            // 在每个盘上指派任务
            for (int block = 0; block < object_size; ++block)
            {
                pair<int, int> task = { object_id, block };

                if (full_req[object_id][block].size() != 0)
                {
                    for (auto disk_index : full_req[object_id][block])
                    {
                        disks[disk_index].assign_task(task);
                    }
                    continue;
                }

                for (int disk_index : objects[object_id].replica)
                {
                    full_req[object_id][block].emplace_back(disk_index);
                    disks[disk_index].assign_task(task);
                }
            }
        }
    }


    void remove_busy_req() {
        for (int disk_index = 0; disk_index < disk_num; disk_index++) {
            disks[disk_index].update_step_predict();
        }


        unordered_map<int, int> expired_requests;

        unordered_map<int, double> cached_result;

        // 从链表头部开始遍历
        RequestNode* current = request_manager.head->next;
        while (current != request_manager.tail) {
            int object_id = current->obj_id;


            if (cached_result.find(object_id) == cached_result.end()) {
                double min_steps = 100000;

                for (auto disk_index : objects[object_id].replica) {
                    double steps = disks[disk_index].predict_complete_time(make_pair(object_id, objects[object_id].size - 1));
                    min_steps = min(min_steps, steps);
                }

                cached_result[object_id] = min_steps;
            }

            double min_steps = cached_result[object_id];

            int rest_steps = 104 - (timestamp - current->timestamp);

            if (rest_steps > 80) {
                break;
            }
            // if ( min_steps >= rest_steps * (1 + tags[objects[object_id].tag].read_times[int(timestamp / 1800)] / 2000000.0))
            if (min_steps >= rest_steps * (1 + 2 * rest_steps / 104.0)) {
                expired_requests[object_id] += 1;

                // 记录下一个节点
                RequestNode* next_node = current->next;

                // 从链表中移除当前节点
                current->prev->next = current->next;
                current->next->prev = current->prev;

                // 从哈希表中移除
                request_manager.map.erase(current->id);

                // 释放当前节点内存
                delete current;

                // 移动到下一个节点
                current = next_node;
            }
            else {
                current = current->next;
            }

        }

        for (auto item = expired_requests.begin(); item != expired_requests.end(); item++) {
            int expired_object_id = item->first;
            int expired_count = item->second;
            process_expired_object(expired_object_id, expired_count);
        }

    }


    void read_action(const vector<pair<int, int>>& read_data)
    {
        // 指派任务
        assign_tasks(read_data);

        vector<int> complete_req;
        for (Disk& disk : disks)
        {
            auto complete_tasks = disk.work();

            for (const auto& task : complete_tasks)
            {
                if (task.first == NONE)
                    continue;

                int object_id = task.first;
                int block = task.second;
                int tag = objects[object_id].size;

                // 取消磁盘上已经完成的任务
                for (int disk_index : full_req[object_id][block])
                {
                    disks[disk_index].cancel_task(task);
                }

                full_req[object_id][block].clear();

                // 更新请求的完成情况，并删除已完成的请求
                vector<pair<int, set<int>>> tmp_remove;
                for (auto& req : req_object_ids[object_id])
                {
                    if (req.second.count(block))
                    {
                        req.second.erase(block);
                    }
                    if (req.second.empty())
                    {
                        tmp_remove.emplace_back(req);
                        complete_req.emplace_back(req.first);
                        request_manager.removeRequest(req.first);
                    }
                }
                for (const auto& req : tmp_remove)
                {
                    auto& reqs = req_object_ids[object_id];
                    reqs.erase(remove(reqs.begin(), reqs.end(), req), reqs.end());
                }
            }
        }

        // 计算每个物品上过期请求的数量
        unordered_map<int, int> expired_requests = request_manager.ExpiredRequests(timestamp);
        for (auto item = expired_requests.begin(); item != expired_requests.end(); item++) {
            int expired_object_id = item->first;
            int expired_count = item->second;
            process_expired_object(expired_object_id, expired_count);
        }


        if (timestamp % 20 == 0) {
            remove_busy_req();
        }


        printf("%zu\n", complete_req.size());
        for (int id : complete_req) {
            printf("%d\n", id);
        }

        // 输出过期请求
        printf("%zu\n", busy_req.size());
        for (int num : busy_req) {
            printf("%d\n", num);
        }
        busy_req.clear();

        fflush(stdout);
    }

    pair<vector<int>, vector<int>> disk_bad_pages_gc(int disk_index, int remain_exchange) {
        // 回收与页标签不符的数据
        vector<set<int>> bad_pages;

        // 收集每个页中不规范物体的数量
        for (int page_index = 0; page_index < disks[disk_index].max_pages; page_index++) {
            int page_start = disks[disk_index].pages[page_index].first;
            int page_end = disks[disk_index].pages[page_index].second;

            set<int> bad_objects;
            for (int index = page_start; index <= page_end; index++) {
                int object_id = disks[disk_index].data[index].first;
                if (object_id == NONE) {
                    continue;
                }

                int object_tag = objects[object_id].tag;
                if (object_tag != disks[disk_index].pages_occupy[page_index]) {
                    bad_objects.insert(object_id);
                }
            }

            bad_pages.emplace_back(bad_objects);
        }

        // 创建一个辅助数组，存储每个 set 的大小及其原始下标
        vector<pair<int, int>> indexed_sizes; // pair: (set大小, 原始下标)
        for (int i = 0; i < bad_pages.size(); ++i) {
            indexed_sizes.emplace_back(bad_pages[i].size(), i);
        }

        // 按照 set 的大小进行排序（由小到大）
        sort(indexed_sizes.begin(), indexed_sizes.end(), [](const auto& a, const auto& b) {
            return a.first < b.first; // 按第一个元素（set大小）排序
            });

        // int remain_exchange = exchange_num;
        vector<int> ori_replace;
        vector<int> new_replace;

        for (const auto& entry : indexed_sizes) {
            int ori_page_index = entry.second;

            for (int object_id : bad_pages[ori_page_index]) {
                int object_size = objects[object_id].size;
                // 超出搬运上限，结束
                // 也许剩余的交换量足以换一个小的物体，这里可以考虑使用continue
                if (object_size > remain_exchange) {
                    continue;
                }

                // 存储符合条件的下标及其对应的pages_volume值
                vector<pair<int, int>> indexed_volumes;

                // 遍历pages_occupy和pages_volume，筛选出pages_occupy等于tag的位置
                for (int i = 0; i < disks[disk_index].pages_occupy.size(); ++i) {
                    if (disks[disk_index].pages_occupy[i] == objects[object_id].tag) {
                        indexed_volumes.emplace_back(i, disks[disk_index].pages_volume[i]);
                    }
                }

                // 按照pages_volume的值进行升序排序
                sort(indexed_volumes.begin(), indexed_volumes.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                    });

                int avail_page_index = NONE;
                for (const auto& entry : indexed_volumes) {
                    if (entry.second >= object_size) {
                        avail_page_index = entry.first;
                        break;
                    }
                }
                if (avail_page_index == NONE) {
                    continue;
                }

                // 启动整理，修改一系列数据
                vector<int> ori_unit;
                int replica_index = NONE;
                for (int i = 0; i < 3; i++) {
                    if (objects[object_id].replica[i] == disk_index) {
                        replica_index = i;
                        ori_unit = objects[object_id].unit[i];
                        break;
                    }
                }

                vector<int> new_unit;
                new_unit = disks[disk_index].transfer_to_page(object_id, object_size, avail_page_index, ori_unit, ori_page_index);
                objects[object_id].unit[replica_index] = new_unit;

                remain_exchange -= object_size;

                ori_replace.insert(ori_replace.end(), ori_unit.begin(), ori_unit.end());
                new_replace.insert(new_replace.end(), new_unit.begin(), new_unit.end());

            }
        }

        return { ori_replace, new_replace };
    }

    pair<vector<int>, vector<int>> disk_small_pages_gc(int disk_index, int remain_exchange) {
        // 回收与页标签相符的数据
        vector<set<int>> small_pages;

        // 收集每个页中规范物体的数量
        for (int page_index = 0; page_index < disks[disk_index].max_pages; page_index++) {
            int page_start = disks[disk_index].pages[page_index].first;
            int page_end = disks[disk_index].pages[page_index].second;

            set<int> small_objects;
            for (int index = page_start; index <= page_end; index++) {
                int object_id = disks[disk_index].data[index].first;
                if (object_id == NONE) {
                    continue;
                }

                int object_tag = objects[object_id].tag;
                if (object_tag == disks[disk_index].pages_occupy[page_index]) {
                    small_objects.insert(object_id);
                }
            }

            small_pages.emplace_back(small_objects);
        }

        // 创建一个辅助数组，存储每个 set 的大小及其原始下标
        vector<pair<int, int>> indexed_sizes; // pair: (set大小, 原始下标)
        for (int i = 0; i < small_pages.size(); ++i) {
            indexed_sizes.emplace_back(small_pages[i].size(), i);
        }

        // 按照 set 的大小进行排序（由小到大）
        sort(indexed_sizes.begin(), indexed_sizes.end(), [](const auto& a, const auto& b) {
            return a.first < b.first; // 按第一个元素（set大小）排序
            });

        // int remain_exchange = exchange_num;
        vector<int> ori_replace;
        vector<int> new_replace;

        for (const auto& entry : indexed_sizes) {
            int ori_page_index = entry.second;

            for (int object_id : small_pages[ori_page_index]) {
                int object_size = objects[object_id].size;
                // 超出搬运上限，结束
                // 也许剩余的交换量足以换一个小的物体，这里可以考虑使用continue
                if (object_size > remain_exchange) {
                    continue;
                }

                // 存储符合条件的下标及其对应的pages_volume值
                vector<pair<int, int>> indexed_volumes;

                // 遍历pages_occupy和pages_volume，筛选出pages_occupy等于tag的位置
                for (int i = 0; i < disks[disk_index].pages_occupy.size(); ++i) {
                    if (disks[disk_index].pages_occupy[i] == objects[object_id].tag) {
                        indexed_volumes.emplace_back(i, disks[disk_index].pages_volume[i]);
                    }
                }

                // 按照pages_volume的值进行升序排序
                sort(indexed_volumes.begin(), indexed_volumes.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                    });

                int avail_page_index = NONE;
                for (const auto& entry : indexed_volumes) {
                    if (entry.second >= object_size) {
                        avail_page_index = entry.first;
                        break;
                    }
                }
                if (avail_page_index == NONE || avail_page_index == ori_page_index) {
                    continue;
                }

                // 启动整理，修改一系列数据
                vector<int> ori_unit;
                int replica_index = NONE;
                for (int i = 0; i < 3; i++) {
                    if (objects[object_id].replica[i] == disk_index) {
                        replica_index = i;
                        ori_unit = objects[object_id].unit[i];
                        break;
                    }
                }

                vector<int> new_unit;
                new_unit = disks[disk_index].transfer_to_page(object_id, object_size, avail_page_index, ori_unit, ori_page_index);
                objects[object_id].unit[replica_index] = new_unit;

                remain_exchange -= object_size;

                ori_replace.insert(ori_replace.end(), ori_unit.begin(), ori_unit.end());
                new_replace.insert(new_replace.end(), new_unit.begin(), new_unit.end());

            }
        }

        return { ori_replace, new_replace };
    }

    pair<vector<int>, vector<int>> disk_pages_self_gc(int disk_index, int remain_exchange) {
        // 页内自整理

        // 创建一个辅助数组，存储每个页的大小及其原始下标
        vector<pair<int, int>> indexed_sizes; // pair: (set大小, 原始下标)
        for (int i = 0; i < disks[disk_index].max_pages; ++i) {
            indexed_sizes.emplace_back(disks[disk_index].pages_volume[i], i);
        }

        // 按照页的剩余空间进行排序（由大到小）
        sort(indexed_sizes.begin(), indexed_sizes.end(), [](const auto& a, const auto& b) {
            return a.first > b.first; // 按第一个元素排序
            });

        // int remain_exchange = exchange_num;
        vector<int> ori_replace;
        vector<int> new_replace;

        for (const auto& entry : indexed_sizes) {
            int page_index = entry.second;
            int page_start = disks[disk_index].pages[page_index].first;
            int page_end = disks[disk_index].pages[page_index].second;

            // 收集每个物品最后一个块的位置
            unordered_map<int, int> page_objects;
            for (int index = page_start; index <= page_end; index++) {
                int object_id = disks[disk_index].data[index].first;
                if (object_id == NONE) {
                    continue;
                }
                // 对于同一个页中不同tag的数据，要不要分别进行处理？
                page_objects[object_id] = index;
            }

            // 将 unordered_map 的元素拷贝到 vector 中
            std::vector<std::pair<int, int>> vec(page_objects.begin(), page_objects.end());

            // 使用 std::sort 对 vector 按值从大到小排序
            std::stable_sort(vec.begin(), vec.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second > b.second; // 按值（second）降序排序
                });

            for (const auto& item : vec) {
                int object_id = item.first;
                int last_block_index = item.second;
                int object_size = objects[object_id].size;

                if (object_size > remain_exchange) {
                    continue;
                }

                // 对于已经在规划下的数据，不迁移
                if (disks[disk_index].index_mask[last_block_index] != 0) {
                    continue;
                }

                vector<int> tmp_position = disks[disk_index].find_write_space(object_size, page_index);

                // 对于申请不到空间的，或是申请的空间较为靠后的，不迁移
                if (tmp_position[tmp_position.size() - 1] == NONE || tmp_position[tmp_position.size() - 1] >= last_block_index) {
                    continue;
                }

                // 启动整理，修改一系列数据
                vector<int> ori_unit;
                int replica_index = NONE;
                for (int i = 0; i < 3; i++) {
                    if (objects[object_id].replica[i] == disk_index) {
                        replica_index = i;
                        ori_unit = objects[object_id].unit[i];
                        break;
                    }
                }

                disks[disk_index].transfer_to_unit(tmp_position, object_id, ori_unit);

                objects[object_id].unit[replica_index] = tmp_position;

                remain_exchange -= object_size;

                ori_replace.insert(ori_replace.end(), ori_unit.begin(), ori_unit.end());
                new_replace.insert(new_replace.end(), tmp_position.begin(), tmp_position.end());

            }
        }

        return { ori_replace, new_replace };
    }

    void disk_gc(int disk_index) {


        pair<vector<int>, vector<int>> bad_result = disk_bad_pages_gc(disk_index, exchange_num);
        vector<int> ori_replace = bad_result.first;
        vector<int> new_replace = bad_result.second;

        if (ori_replace.size() < exchange_num) {
            pair<vector<int>, vector<int>> self_result = disk_pages_self_gc(disk_index, exchange_num - ori_replace.size());
            vector<int> self_result_ori_replace = self_result.first;
            vector<int> self_result_new_replace = self_result.second;

            ori_replace.insert(ori_replace.end(), self_result_ori_replace.begin(), self_result_ori_replace.end());
            new_replace.insert(new_replace.end(), self_result_new_replace.begin(), self_result_new_replace.end());

            /*
            pair<vector<int>, vector<int>> small_result = disk_small_pages_gc(disk_index, exchange_num - ori_replace.size());
            vector<int> small_result_ori_replace = small_result.first;
            vector<int> small_result_new_replace = small_result.second;

            ori_replace.insert(ori_replace.end(), small_result_ori_replace.begin(), small_result_ori_replace.end());
            new_replace.insert(new_replace.end(), small_result_new_replace.begin(), small_result_new_replace.end());
            */
        }


        // 输出
        printf("%zu\n", ori_replace.size());
        for (int i = 0; i < ori_replace.size(); i++) {
            printf("%d %d\n", ori_replace[i] + 1, new_replace[i] + 1);
        }
    }

    void gc_action()
    {
        printf("GARBAGE COLLECTION\n");
        for (int i = 0; i < disk_num; i++) {
            // printf("0\n");
            disk_gc(i);
        }
        fflush(stdout);
    }
};


/*
// 本地调试用代码，直接从本地读取数据
class InputReader {
public:
    vector<string> lines;
    size_t index = 0;

    InputReader() {

        string file_path = "data_path\\sample_practice.in";
        ifstream file(file_path);
        string line;
        while (getline(file, line)) {
            if (!line.empty()) lines.push_back(line);
        }
    }

    string next_line() {
        return index < lines.size() ? lines[index++] : "";
    }

    int next_int() {
        return stoi(next_line());
    }

    vector<string> split(const string& s) {
        stringstream ss(s);
        string token;
        vector<string> result;
        while (ss >> token) result.push_back(token);
        return result;
    }

    string timestamp_input() {
        return split(next_line())[1];
    }

    vector<int> delete_input() {
        int delete_num = next_int();
        vector<int> ids(delete_num);
        for (int i = 0; i < delete_num; ++i) ids[i] = next_int();
        return ids;
    }

    vector<tuple<int, int, int>> write_input() {
        int write_num = next_int();
        vector<tuple<int, int, int>> data;
        for (int i = 0; i < write_num; ++i) {
            auto item = split(next_line());
            int id = stoi(item[0]), size = stoi(item[1]), tag = stoi(item[2]) - 1;
            data.emplace_back(id, size, tag);
        }
        return data;
    }

    vector<pair<int, int>> read_input() {
        int read_num = next_int();
        vector<pair<int, int>> data;
        for (int i = 0; i < read_num; ++i) {
            auto item = split(next_line());
            int rid = stoi(item[0]), oid = stoi(item[1]);
            data.emplace_back(rid, oid);
        }
        return data;
    }

    tuple<int, int, int, int, int, int, vector<vector<int>>, vector<vector<int>>, vector<vector<int>>>
        read_initial_data() {
        auto base = split(next_line());
        int T = stoi(base[0]), M = stoi(base[1]), N = stoi(base[2]), V = stoi(base[3]), G = stoi(base[4]), K = stoi(base[5]);
        vector<vector<int>> del(M), wr(M), rd(M);
        for (int i = 0; i < M; ++i) del[i] = parse_line_of_ints();
        for (int i = 0; i < M; ++i) wr[i] = parse_line_of_ints();
        for (int i = 0; i < M; ++i) rd[i] = parse_line_of_ints();
        return { T, M, N, V, G, K, del, wr, rd };
    }

    void gc_input()
    {
        // 直接跳过该行，不需要处理
        next_line();
    }

private:
    vector<int> parse_line_of_ints() {
        auto items = split(next_line());
        vector<int> result;
        for (auto& s : items) result.push_back(stoi(s));
        return result;
    }
};
*/



class InputReader
{
public:
    InputReader() : buffer_index(0) {}

    string next_line()
    {
        if (buffer_index < buffer.size())
            return buffer[buffer_index++];
        string line;
        if (getline(cin, line))
        {
            buffer.push_back(line);
            buffer_index++;
            return line;
        }
        return "";
    }

    int next_int()
    {
        string line = next_line();
        return line.empty() ? 0 : stoi(line);
    }

    vector<string> split(const string& s)
    {
        stringstream ss(s);
        vector<string> result;
        string token;
        while (ss >> token)
            result.emplace_back(move(token));
        return result;
    }

    string timestamp_input()
    {
        string line = next_line();
        size_t pos = line.find(' ');
        return pos != string::npos ? line.substr(pos + 1) : "";
    }

    vector<int> delete_input()
    {
        int delete_num = next_int();
        vector<int> ids(delete_num);
        for (int i = 0; i < delete_num; ++i)
            ids[i] = next_int();
        return ids;
    }

    vector<tuple<int, int, int>> write_input()
    {
        int write_num = next_int();
        vector<tuple<int, int, int>> data;
        data.reserve(write_num); // 预分配空间
        for (int i = 0; i < write_num; ++i)
        {
            string line = next_line();
            size_t pos1 = line.find(' '), pos2 = line.find(' ', pos1 + 1);
            int id = stoi(line.substr(0, pos1));
            int size = stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
            int tag = stoi(line.substr(pos2 + 1)) - 1;
            data.emplace_back(id, size, tag);
        }
        return data;
    }

    vector<pair<int, int>> read_input()
    {
        int read_num = next_int();
        vector<pair<int, int>> data;
        data.reserve(read_num); // 预分配空间
        for (int i = 0; i < read_num; ++i)
        {
            string line = next_line();
            size_t pos = line.find(' ');
            int rid = stoi(line.substr(0, pos));
            int oid = stoi(line.substr(pos + 1));
            data.emplace_back(rid, oid);
        }
        return data;
    }

    tuple<int, int, int, int, int, int, vector<vector<int>>, vector<vector<int>>, vector<vector<int>>, vector<int>>
        read_initial_data()
    {
        string line = next_line();
        size_t pos1 = line.find(' '), pos2 = line.find(' ', pos1 + 1),
            pos3 = line.find(' ', pos2 + 1), pos4 = line.find(' ', pos3 + 1),
            pos5 = line.find(' ', pos4 + 1);
        int T = stoi(line.substr(0, pos1));
        int M = stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
        int N = stoi(line.substr(pos2 + 1, pos3 - pos2 - 1));
        int V = stoi(line.substr(pos3 + 1, pos4 - pos3 - 1));
        int G = stoi(line.substr(pos4 + 1, pos5 - pos4 - 1));
        int K = stoi(line.substr(pos5 + 1));

        vector<vector<int>> del(M), wr(M), rd(M);
        for (int i = 0; i < M; ++i)
            del[i] = parse_line_of_ints();
        for (int i = 0; i < M; ++i)
            wr[i] = parse_line_of_ints();
        for (int i = 0; i < M; ++i)
            rd[i] = parse_line_of_ints();

        // 读取g[i]
        vector<int> g = parse_line_of_ints();

        return { T, M, N, V, G, K, move(del), move(wr), move(rd), move(g) };
    }

    // 垃圾回收输入，不需要返回值
    void gc_input()
    {
        next_line();
    }

private:
    vector<int> parse_line_of_ints()
    {
        string line = next_line();
        vector<int> result;
        size_t start = 0, end;
        while ((end = line.find(' ', start)) != string::npos)
        {
            result.emplace_back(stoi(line.substr(start, end - start)));
            start = end + 1;
        }
        if (start < line.size())
            result.emplace_back(stoi(line.substr(start)));
        return result;
    }

    vector<string> buffer;
    size_t buffer_index;
};



int main()
{
    InputReader reader;

    int T, M, N, V, G, K;
    vector<vector<int>> delete_data, write_data, read_data;
    vector<int> g;
    tie(T, M, N, V, G, K, delete_data, write_data, read_data, g) = reader.read_initial_data();

    vector<TagMessage> tag_list;
    for (int i = 0; i < M; ++i)
    {
        TagMessage tag(i, delete_data[i], write_data[i], read_data[i]);
        tag_list.emplace_back(tag);
    }

    DiskManager manager(T, M, N, V, G, K, tag_list, g);

    for (int t = 0; t < T + EXTRA_TIME; ++t)
    {
        string timestamp = reader.timestamp_input();
        manager.timestamp_action(stoi(timestamp));

        auto delete_ids = reader.delete_input();
        manager.delete_action(delete_ids);

        auto write_ops = reader.write_input();
        manager.write_action(write_ops);

        auto read_ops = reader.read_input();
        manager.read_action(read_ops);

        if ((t + 1) % FRE_PER_SLICING == 0) {
            reader.gc_input();
            manager.gc_action();
        }
    }

    return 0;
}