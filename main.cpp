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

// pair<int, int>�Ĺ�ϣ����
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

// �����ֵ
double compute_mean(const vector<int>& l)
{
    double sum = 0.0;
    for (int v : l)
        sum += v;
    return sum / l.size();
}

// ���������ȳ��б��Ƥ��ѷ���ϵ��
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

// ����Ƥ��ѷ���ϵ������
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

// ��������ʱ�����еķ�ֵ(���ֵ)λ�ü������ϵ������ӳ�䷶ΧΪ[-1,1]
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

// ������ƽ����Ϊk��ҳ����ÿ����ҳ����ʼ�ͽ����±�洢��һ��vector��
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

// һ��ʱ�������ڣ�����ʼ��token cost��ʼ������������ָ��˥���ķ������������һֱ�����
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

// �洢��Ԫ���࣬��Ÿõ�Ԫ��id����С����ǩ�������ʹ洢λ��
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

// ��ǩ���࣬��Ÿñ�ǩ��id��ɾ��������д������Ͷ�ȡ������Ϣ
class TagMessage
{
public:
    int tag;
    vector<int> delete_times;
    vector<int> write_times;
    vector<int> read_times;

    vector<pair<int, int>> space;         // tag��ӵ�е�ҳ

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
    int name;                                       // �̺�
    int volume;                                     // ��������
    int tokens;                                     // ÿ�غϿ���ʹ�õ�token
    vector<pair<int, int>> data;                    // ���д洢������
    unordered_map<pair<int, int>, int> index;       // �ֵ䣬������������λ��
    int head[2];                                    // ��ͷλ��
    int read_length;                                // �ܹ���ȡ����Զ����
    int last_read_token[2];                         // ��¼�ϴζ��ж�ʹ�õ�token��������ϴβ��Ƕ��ж�����ΪNONE

    int mask_count[2];                              // ÿ����ͷռ�е�mask����
    int work_page[2];                               // ÿ����ͷ�Ĺ���ҳ

    int timestamp = 0;                              // ��ǰʱ��Ƭ
    int time_period = 0;                            // ��ǰ���ڣ� ��1800ʱ��ƬΪ1����
    vector<int> tasks_count;                        // ��¼������̲�ͬλ�����ݵ�������
    unordered_map<pair<int, int>, int>& task_mask;  // ������disk manager���������ݵ�ռ�����
    vector<int> index_mask;                         // ��ǰ������ÿ���±��ռ�ã�����Ӧ��������д���ɾ�������

    vector<int> page_load;                          // ÿ��ҳ����Ҫ������������
    vector<double> step_predict;                    // ÿ��ҳ��Ԥ�����ʱ��

    vector<vector<int>> read_times;                 // ÿ��tag��ÿ�����ڵĶ�ȡ��

    int max_pages;                                  // ��������ҳ����
    vector<int> pages_occupy;                       // ÿ����ҳ�ϴ洢��tag�±�
    vector<pair<int, int>> pages;                   // ÿ����ҳ����ʼ�ͽ����±�
    vector<int> pages_volume;                       // ÿ��ҳ��ʣ������
    int base_volume;                                // ÿ��ҳ�Ļ������ȣ����ڼ����±�������ҳ

    int max_tag_num;                                // ���tag��(legacy)
    vector<vector<double>> similarity;              // ��ͬtag��ȡ�����Ļ��ڷ�ֵ�����ƶȣ�����д�����еķ�ҳѡ��(legacy)
    vector<vector<double>> similarity_for_disk;     // ��ͬtag��ȡ������Ƥ��ѷ���ƶȣ�����д�����еķ���ѡ��
    double base_ability;                            // ������̵Ļ�����ȡ����

    vector<int> read_plan;                          // ���ݵ�ǰ�洢��tag������ÿ�����ڵĶ�ȡ��(legacy)
    vector<vector<vector<double>>> similarity_decay;// �����˥�������ƶ�
    vector<vector<double>> position_value;          // ÿ��λ�����ƶȵ��ܺ�(legacy)

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
        // ��ʼ��������
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

    // ָ������
    void assign_task(const pair<int, int>& task)
    {
        int task_index = index[task];
        // tasks_queue[104][task_index]++;
        tasks_count[task_index] += 1;

        if (tasks_count[task_index] == 1) {
            page_load[min(task_index / base_volume, max_pages - 1)] += 1;
        }

    }

    // ȡ������
    void cancel_task(const pair<int, int>& task)
    {
        int task_index = index[task];
        // completed_tasks[task_index] += tasks_count[task_index];
        if (tasks_count[task_index] != 0) {
            page_load[min(task_index / base_volume, max_pages - 1)] -= 1;
        }
        tasks_count[task_index] = 0;
    }

    // ȡ���ض�����������
    void cancel_expired_task(const pair<int, int>& task, int cancel_num)
    {
        int task_index = index[task];
        // completed_tasks[task_index] += tasks_count[task_index];
        tasks_count[task_index] -= cancel_num;
        if (tasks_count[task_index] == 0) {
            page_load[min(task_index / base_volume, max_pages - 1)] -= 1;
        }
    }

    // ������
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

                // ���������Ѿ���ɵ��������tasks count
                if (tasks_count[head[head_index]] != 0) {
                    page_load[min(head[head_index] / base_volume, max_pages - 1)] -= 1;
                }
                tasks_count[head[head_index]] = 0;
            }
            else
            {
                // �������Ҫ��������ݺ����ļ�ֵ�жϵ�ǰ�Ƿ�Ҫ������
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
                    // ����ʣ���token���ڵ�ǰ�Ķ�cost�»��ܶ����ٲ�
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

    // ִ������
    vector<pair<int, int>> work()
    {
        vector<pair<int, int>> complete_tasks;

        // ������������
        int max_range = min(tokens - 64, volume - 1);

        for (int head_index = 0; head_index < 2; head_index++) {
            int remain_token = tokens;

            while (true) {
                // ���û�й������ͷ���һ������
                if (work_page[head_index] == NONE) {
                    get_mask(head_index);
                }

                // ��������˹��������ǲ��ڶ�Ӧ��ҳ�ڣ�����취����ҳ����ʼλ��
                int page_index = work_page[head_index];
                if (min(head[head_index] / base_volume, max_pages - 1) != page_index) {
                    last_read_token[head_index] = NONE;

                    int distance = (pages[page_index].first - head[head_index] + volume) % volume;

                    if (distance <= remain_token) {
                        // �����ֱ��p��ȥ����p��ȥ
                        for (int i = 0; i < distance; i++) {
                            printf("p");
                        }
                        head[head_index] = (head[head_index] + distance) % volume;
                        remain_token -= distance;
                    }
                    else if (remain_token == tokens) {
                        // �������p��ȥ���ͳ�����Ծ��ȥ
                        printf("j %d\n", pages[page_index].first + 1);
                        head[head_index] = pages[page_index].first;
                        remain_token = 0;
                        break;
                    }
                    else {
                        // ���Ҳ��������ȥ���;�����pass
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

    // ���ݼ�ֵѡ��һ����õ�ҳ������ѡ��û�б�ռ�ݵ�ҳ
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

    // ����ҳ�����޸ļ�ֵ����
    pair<int, int> malloc_space(int tag)
    {
        // // ��ת�������߼�����Ϊ�����Ƶı�ǩ�ֿ�
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

    // ֻ����ռ䣬��д������
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

    // ֱ��д��unit
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

    // ������д���ض���ҳ��
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

    // ɾ���ض�unit������
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

    // ת�����ݣ�ת��ʱ��ֻ����Ŀ��ҳ������λ���ɴ�����������
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

    // ֱ��д�뵽ָ��λ��
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

    // �ô��̵�����ֵ�����ڴ���ѡ��
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

    // ����һ������λ���ĸ�ҳ���Լ���ҳ�ڵ�ƫ����
    pair<int, int> index_position(int index) {
        int page_index = min(index / base_volume, max_pages - 1);
        int offset = index - page_index * base_volume;

        return { page_index, offset };
    }

    // �����ڵ�ǰ״���£�ĳ���±��������Ҫ�ȴ���ʱ��
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

    // ����һ��ҳ��û�б�ռ�е����ݼ�ֵ
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

    // ����ҳ��ֵѡ��ҳ
    int best_masked_page(int head_index) {
        int head_page = min(head[head_index] / base_volume, max_pages - 1);

        double best_value = NONE;
        int best_index = NONE;

        for (int page_index = 0; page_index < max_pages; page_index++) {
            double page_value = masked_page_value(page_index);
            if (page_index == head_page) {
                // �ڽ���ҳ����buff
                page_value *= 1.2;
            }

            if (page_value > best_value) {
                best_value = page_value;
                best_index = page_index;
            }
        }
        return best_index;
    }

    // Ϊ��ͷѡ��ҳ������ȡmask��ͬ����disk manager
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

    // ����ÿ��ҳ֮ǰ��Ԥ�ڲ���
    void update_step_predict() {
        vector<pair<int, double>> page_value;
        for (int page_index = 0; page_index < max_pages; page_index++) {
            double value = masked_page_value(page_index);
            page_value.emplace_back(make_pair(page_index, value));
        }

        std::stable_sort(page_value.begin(), page_value.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second; // ��ֵ��second����������
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

    // ����һ���������ɲ���
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


// ��������ڵ�
struct RequestNode {
    int id;              // ���� ID
    int timestamp;       // ʱ���
    int obj_id;          // ���� ID
    RequestNode* prev;   // ǰһ������
    RequestNode* next;   // ��һ������

    // ���캯��
    RequestNode(int _id, int _timestamp, int _obj_id)
        : id(_id), timestamp(_timestamp), obj_id(_obj_id), prev(nullptr), next(nullptr) {
    }
};

// �������������
class RequestManager {
public:
    RequestNode* head;                         // ����ͷ�ڱ��ڵ�
    RequestNode* tail;                         // ����β�ڱ��ڵ�
    unordered_map<int, RequestNode*> map; // ��ϣ������ͨ�� ID ���ٲ�������

    // ���캯��
    RequestManager() {
        head = new RequestNode(NONE, NONE, NONE); // ͷ�ڱ��ڵ㣬ID��timestamp �� obj_id Ϊ��Чֵ
        tail = new RequestNode(NONE, NONE, NONE); // β�ڱ��ڵ㣬ID��timestamp �� obj_id Ϊ��Чֵ
        head->next = tail;
        tail->prev = head;
    }

    // �����������ͷ����ж�̬������ڴ�
    ~RequestManager() {
        RequestNode* current = head;
        while (current) {
            RequestNode* nextNode = current->next;
            delete current;
            current = nextNode;
        }
    }

    // ��������
    void addRequest(int id, int timestamp, int obj_id) {
        // �����½ڵ�
        RequestNode* newNode = new RequestNode(id, timestamp, obj_id);

        // ���뵽����ĩβ��β�ڱ��ڵ�֮ǰ��
        newNode->prev = tail->prev;
        newNode->next = tail;
        tail->prev->next = newNode;
        tail->prev = newNode;

        // ���¹�ϣ��
        map[id] = newNode;
    }

    // ɾ������
    void removeRequest(int req_id) {

        // ��ȡ����ڵ�
        RequestNode* node = map[req_id];

        // ���������Ƴ��ڵ�
        node->prev->next = node->next;
        node->next->prev = node->prev;

        // �ӹ�ϣ�����Ƴ�
        map.erase(req_id);

        // �ͷŽڵ��ڴ�
        delete node;
    }

    // ��ȡ��������ͳ�Ʋ�ɾ����������
    unordered_map<int, int> ExpiredRequests(int current_timestamp) {
        unordered_map<int, int> expired_count; // �洢 obj_id �ļ���
        int max_time = 104; // ����104ʱ��Ƭû����ɵģ�ֱ���ϱ���æ

        // ������ͷ����ʼ����
        RequestNode* current = head->next;
        while (current != tail) {
            if (current_timestamp - current->timestamp > max_time) {
                // ͳ�� obj_id
                expired_count[current->obj_id]++;

                // ��¼��һ���ڵ�
                RequestNode* next_node = current->next;

                // ���������Ƴ���ǰ�ڵ�
                current->prev->next = current->next;
                current->next->prev = current->prev;

                // �ӹ�ϣ�����Ƴ�
                map.erase(current->id);

                // �ͷŵ�ǰ�ڵ��ڴ�
                delete current;

                // �ƶ�����һ���ڵ�
                current = next_node;
            }
            else {
                // һ������δ���ڵĽڵ㣬ֹͣ����
                break;
            }
        }

        return expired_count;
    }

};



class DiskManager
{
public:
    int PAGES_NUM = 20;             // ÿ�����̵�ҳ�������ֵ����20-24֮��
    int timestamp;                  // ��ǰʱ��Ƭ
    int time_num;                   // ��ʱ��Ƭ����
    int tag_num;                    // ��tag����
    int disk_num;                   // �ܴ�������
    int disk_volume;                // ���̵�����
    int disk_tokens;                // ���̿���ʹ�õ�token��
    int exchange_num;               // ÿ���������տ��Խ����Ĵ���
    vector<int> disk_tokens_list;   // ÿ�����̵�token��

    unordered_map<pair<int, int>, int> task_mask;       // ���ݵ�ռ�����

    vector<TagMessage> tags;        // ÿ��tag����Ϣ

    int rep_num = 3;                // ��������
    int fre_per_slicing = 1800;     // ÿ�����ڵ�ʱ��Ƭ��

    vector<Disk> disks;                                                 // ���̶���
    unordered_map<int, Object> objects;                                 // �ļ�����
    unordered_map<int, deque<pair<int, set<int>>>> req_object_ids;      // ��Ҫ��ȡĳ���ļ������������Լ�ʵʱ������
    vector<vector<vector<int>>> full_req;                               // ���������Կ�Ϊ��λ����ָ�ɵĴ���

    RequestManager request_manager;                                     // һ�ָ�Ч��������������������ṹΪ��ϣ˫������

    vector<int> busy_req;                                               // ��æ������

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

        // �����ȡ��ֵ
        vector<int> all_read_times;
        all_read_times.resize(read_times[0].size(), 0);
        for (int index = 0; index < read_times[0].size(); index++)
        {
            for (auto& tag_read : read_times)
            {
                all_read_times[index] += tag_read[index];
            }
        }

        // �������ƾ���
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

        // ��ʼ�����̶���
        for (int i = 0; i < disk_num; ++i)
        {
            disks.emplace_back(Disk(i, disk_volume, disk_tokens,
                max_pages, similarity_for_disk, similarity, similarity_decay,
                read_times, task_mask));
        }

        // ����full_req�ռ�
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

    // �����������
    void process_expired_object(int expired_object_id, int expired_count) {
        // ����洢
        unordered_map<int, int> element_count;

        // ��ȡ��Ӧ deque ������
        deque<pair<int, set<int>>>& dq = req_object_ids[expired_object_id];

        // ����ǰ k ��
        int count = 0;
        for (auto it = dq.begin(); it != dq.end() && count < expired_count; ++it, ++count) {
            const auto& item = *it;

            busy_req.push_back(item.first);

            // ͳ�� set<int> �е�Ԫ��
            for (int elem : item.second) {
                element_count[elem]++;
            }
        }

        // ɾ��ǰ k ��
        dq.erase(dq.begin(), dq.begin() + expired_count);

        int tag = objects[expired_object_id].tag;

        // �����Ѿ�����������
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

        // ����ÿ�����ϵ�ʱ����Ϣ
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
            // ȡ��ÿ������ÿ�����ϵ�����
            for (int block = 0; block < object_size; ++block)
            {
                pair<int, int> task = { delete_id, block };

                for (auto disk_num : full_req[delete_id][block])
                {
                    disks[disk_num].cancel_task(task);
                }

                full_req[delete_id][block].clear();
            }

            // ���㱻ȡ��������
            for (auto& req : req_object_ids[delete_id])
            {
                abort_req.emplace_back(get<0>(req));
                request_manager.removeRequest(get<0>(req));
            }
            req_object_ids.erase(delete_id);

            // ɾ������
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

    // ����ռ䣬��������ѡ����̺�ҳ
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

    // ������������Ǿ�����д��һ��ҳ
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

            // ɸѡ����ҳ
            stable_sort(avail_pages.begin(), avail_pages.end(),
                [this](const pair<int, int>& a, const pair<int, int>& b)
                {
                    return disks[a.first].page_volume(a.second) < disks[b.first].page_volume(b.second);
                });

            int copy_num = 0;
            set<int> write_disks;
            vector<pair<int, vector<int>>> write_history;

            // ����д��
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

            // ������и���û��д�룬�������µ�ҳ
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


            // �Ǽ����������������
            pair<int, set<int>> req_info;
            req_info.first = request_id;
            set<int> block_set;
            for (int i = 0; i < object_size; ++i)
                block_set.insert(i);
            req_info.second = move(block_set);
            req_object_ids[object_id].emplace_back(move(req_info));

            // ��ÿ������ָ������
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

        // ������ͷ����ʼ����
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

                // ��¼��һ���ڵ�
                RequestNode* next_node = current->next;

                // ���������Ƴ���ǰ�ڵ�
                current->prev->next = current->next;
                current->next->prev = current->prev;

                // �ӹ�ϣ�����Ƴ�
                request_manager.map.erase(current->id);

                // �ͷŵ�ǰ�ڵ��ڴ�
                delete current;

                // �ƶ�����һ���ڵ�
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
        // ָ������
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

                // ȡ���������Ѿ���ɵ�����
                for (int disk_index : full_req[object_id][block])
                {
                    disks[disk_index].cancel_task(task);
                }

                full_req[object_id][block].clear();

                // �������������������ɾ������ɵ�����
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

        // ����ÿ����Ʒ�Ϲ������������
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

        // �����������
        printf("%zu\n", busy_req.size());
        for (int num : busy_req) {
            printf("%d\n", num);
        }
        busy_req.clear();

        fflush(stdout);
    }

    pair<vector<int>, vector<int>> disk_bad_pages_gc(int disk_index, int remain_exchange) {
        // ������ҳ��ǩ����������
        vector<set<int>> bad_pages;

        // �ռ�ÿ��ҳ�в��淶���������
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

        // ����һ���������飬�洢ÿ�� set �Ĵ�С����ԭʼ�±�
        vector<pair<int, int>> indexed_sizes; // pair: (set��С, ԭʼ�±�)
        for (int i = 0; i < bad_pages.size(); ++i) {
            indexed_sizes.emplace_back(bad_pages[i].size(), i);
        }

        // ���� set �Ĵ�С����������С����
        sort(indexed_sizes.begin(), indexed_sizes.end(), [](const auto& a, const auto& b) {
            return a.first < b.first; // ����һ��Ԫ�أ�set��С������
            });

        // int remain_exchange = exchange_num;
        vector<int> ori_replace;
        vector<int> new_replace;

        for (const auto& entry : indexed_sizes) {
            int ori_page_index = entry.second;

            for (int object_id : bad_pages[ori_page_index]) {
                int object_size = objects[object_id].size;
                // �����������ޣ�����
                // Ҳ��ʣ��Ľ��������Ի�һ��С�����壬������Կ���ʹ��continue
                if (object_size > remain_exchange) {
                    continue;
                }

                // �洢�����������±꼰���Ӧ��pages_volumeֵ
                vector<pair<int, int>> indexed_volumes;

                // ����pages_occupy��pages_volume��ɸѡ��pages_occupy����tag��λ��
                for (int i = 0; i < disks[disk_index].pages_occupy.size(); ++i) {
                    if (disks[disk_index].pages_occupy[i] == objects[object_id].tag) {
                        indexed_volumes.emplace_back(i, disks[disk_index].pages_volume[i]);
                    }
                }

                // ����pages_volume��ֵ������������
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

                // ���������޸�һϵ������
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
        // ������ҳ��ǩ���������
        vector<set<int>> small_pages;

        // �ռ�ÿ��ҳ�й淶���������
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

        // ����һ���������飬�洢ÿ�� set �Ĵ�С����ԭʼ�±�
        vector<pair<int, int>> indexed_sizes; // pair: (set��С, ԭʼ�±�)
        for (int i = 0; i < small_pages.size(); ++i) {
            indexed_sizes.emplace_back(small_pages[i].size(), i);
        }

        // ���� set �Ĵ�С����������С����
        sort(indexed_sizes.begin(), indexed_sizes.end(), [](const auto& a, const auto& b) {
            return a.first < b.first; // ����һ��Ԫ�أ�set��С������
            });

        // int remain_exchange = exchange_num;
        vector<int> ori_replace;
        vector<int> new_replace;

        for (const auto& entry : indexed_sizes) {
            int ori_page_index = entry.second;

            for (int object_id : small_pages[ori_page_index]) {
                int object_size = objects[object_id].size;
                // �����������ޣ�����
                // Ҳ��ʣ��Ľ��������Ի�һ��С�����壬������Կ���ʹ��continue
                if (object_size > remain_exchange) {
                    continue;
                }

                // �洢�����������±꼰���Ӧ��pages_volumeֵ
                vector<pair<int, int>> indexed_volumes;

                // ����pages_occupy��pages_volume��ɸѡ��pages_occupy����tag��λ��
                for (int i = 0; i < disks[disk_index].pages_occupy.size(); ++i) {
                    if (disks[disk_index].pages_occupy[i] == objects[object_id].tag) {
                        indexed_volumes.emplace_back(i, disks[disk_index].pages_volume[i]);
                    }
                }

                // ����pages_volume��ֵ������������
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

                // ���������޸�һϵ������
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
        // ҳ��������

        // ����һ���������飬�洢ÿ��ҳ�Ĵ�С����ԭʼ�±�
        vector<pair<int, int>> indexed_sizes; // pair: (set��С, ԭʼ�±�)
        for (int i = 0; i < disks[disk_index].max_pages; ++i) {
            indexed_sizes.emplace_back(disks[disk_index].pages_volume[i], i);
        }

        // ����ҳ��ʣ��ռ���������ɴ�С��
        sort(indexed_sizes.begin(), indexed_sizes.end(), [](const auto& a, const auto& b) {
            return a.first > b.first; // ����һ��Ԫ������
            });

        // int remain_exchange = exchange_num;
        vector<int> ori_replace;
        vector<int> new_replace;

        for (const auto& entry : indexed_sizes) {
            int page_index = entry.second;
            int page_start = disks[disk_index].pages[page_index].first;
            int page_end = disks[disk_index].pages[page_index].second;

            // �ռ�ÿ����Ʒ���һ�����λ��
            unordered_map<int, int> page_objects;
            for (int index = page_start; index <= page_end; index++) {
                int object_id = disks[disk_index].data[index].first;
                if (object_id == NONE) {
                    continue;
                }
                // ����ͬһ��ҳ�в�ͬtag�����ݣ�Ҫ��Ҫ�ֱ���д���
                page_objects[object_id] = index;
            }

            // �� unordered_map ��Ԫ�ؿ����� vector ��
            std::vector<std::pair<int, int>> vec(page_objects.begin(), page_objects.end());

            // ʹ�� std::sort �� vector ��ֵ�Ӵ�С����
            std::stable_sort(vec.begin(), vec.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second > b.second; // ��ֵ��second����������
                });

            for (const auto& item : vec) {
                int object_id = item.first;
                int last_block_index = item.second;
                int object_size = objects[object_id].size;

                if (object_size > remain_exchange) {
                    continue;
                }

                // �����Ѿ��ڹ滮�µ����ݣ���Ǩ��
                if (disks[disk_index].index_mask[last_block_index] != 0) {
                    continue;
                }

                vector<int> tmp_position = disks[disk_index].find_write_space(object_size, page_index);

                // �������벻���ռ�ģ���������Ŀռ��Ϊ����ģ���Ǩ��
                if (tmp_position[tmp_position.size() - 1] == NONE || tmp_position[tmp_position.size() - 1] >= last_block_index) {
                    continue;
                }

                // ���������޸�һϵ������
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


        // ���
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
// ���ص����ô��룬ֱ�Ӵӱ��ض�ȡ����
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
        // ֱ���������У�����Ҫ����
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
        data.reserve(write_num); // Ԥ����ռ�
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
        data.reserve(read_num); // Ԥ����ռ�
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

        // ��ȡg[i]
        vector<int> g = parse_line_of_ints();

        return { T, M, N, V, G, K, move(del), move(wr), move(rd), move(g) };
    }

    // �����������룬����Ҫ����ֵ
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