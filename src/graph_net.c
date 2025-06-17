#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>

struct Node {
    double* values;
    double* log_probs;
    int num_messages;
    int bias_index;
    int* weight_indices;
    int num_incoming;
    int* incoming_edge_indices;
};

struct Edge {
    int from;
    int to;
    int filter_param_start;
    int num_filter_params;
    double* values;
    double* log_probs;
    int num_messages;
};

struct Network {
    double* params;
    int params_size;
    struct Node* nodes;
    int num_nodes;
    struct Edge* edges;
    int num_edges;
};

double filter(double value, double* params) {
    double mu = params[0];
    double sigma = params[1];
    return exp(- (value - mu) * (value - mu) / (2 * sigma * sigma));
}

double activation(double x) {
    return x > 0 ? x : 0;
}

void edge_step(struct Network* net) {
    for (int e = 0; e < net->num_edges; e++) {
        struct Edge* edge = &net->edges[e];
        struct Node* source = &net->nodes[edge->from];
        if (edge->values) free(edge->values);
        if (edge->log_probs) free(edge->log_probs);
        edge->num_messages = source->num_messages;
        edge->values = malloc(sizeof(double) * edge->num_messages);
        edge->log_probs = malloc(sizeof(double) * edge->num_messages);
        double* fparams = &net->params[edge->filter_param_start];
        double lost_prob = 0.0;
        for (int i = 1; i < source->num_messages; i++) {
            double f = filter(source->values[i], fparams);
            edge->values[i] = source->values[i];
            edge->log_probs[i] = source->log_probs[i] + log(f);
            lost_prob += exp(source->log_probs[i]) * (1 - f);
        }
        edge->values[0] = 0.0;
        double original_non_event_prob = exp(source->log_probs[0]);
        double new_non_event_prob = original_non_event_prob + lost_prob;
        edge->log_probs[0] = log(new_non_event_prob);
    }
}

void node_step(struct Network* net) {
    for (int n = 0; n < net->num_nodes; n++) {
        struct Node* node = &net->nodes[n];
        int k = node->num_incoming;
        if (k == 0) continue;
        int* sizes = malloc(sizeof(int) * k);
        struct Edge** in_edges = malloc(sizeof(struct Edge*) * k);
        for (int i = 0; i < k; i++) {
            int eidx = node->incoming_edge_indices[i];
            in_edges[i] = &net->edges[eidx];
            sizes[i] = in_edges[i]->num_messages;
        }
        long long total = 1;
        for (int i = 0; i < k; i++) total *= sizes[i];
        double* new_values = malloc(sizeof(double) * total);
        double* new_log_probs = malloc(sizeof(double) * total);
        int* indices = calloc(k, sizeof(int));
        for (long long pos = 0; pos < total; pos++) {
            double sum = net->params[node->bias_index];
            double log_p = 0.0;
            for (int i = 0; i < k; i++) {
                int idx = indices[i];
                struct Edge* edge = in_edges[i];
                sum += net->params[node->weight_indices[i]] * edge->values[idx];
                log_p += edge->log_probs[idx];
            }
            double val = activation(sum);
            new_values[pos] = val;
            new_log_probs[pos] = log_p;
            // increment indices
            int carry = 1;
            for (int i = 0; i < k && carry; i++) {
                indices[i] += carry;
                if (indices[i] >= sizes[i]) {
                    indices[i] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
        }
        free(indices);
        free(sizes);
        free(in_edges);
        if (node->values) free(node->values);
        if (node->log_probs) free(node->log_probs);
        node->values = new_values;
        node->log_probs = new_log_probs;
        node->num_messages = total;
    }
}

void print_node_messages(struct Node* node) {
    for (int i = 0; i < node->num_messages; i++) {
        printf("  value: %f, prob: %f\n", node->values[i], exp(node->log_probs[i]));
    }
}

int main() {
    struct Network net;
    net.num_nodes = 2;
    net.num_edges = 1;
    net.nodes = malloc(sizeof(struct Node) * net.num_nodes);
    net.edges = malloc(sizeof(struct Edge) * net.num_edges);

    // calculate params_size
    int params_size = 0;
    for (int e = 0; e < net.num_edges; e++) params_size += 2; // for filter
    for (int n = 0; n < net.num_nodes; n++) {
        // for simplicity, assign for all, but only use if num_incoming >0
        params_size += 1; // bias
        // weights later
    }
    params_size += 1; // for the weight of node 1

    net.params_size = params_size;
    net.params = malloc(sizeof(double) * params_size);
    int param_idx = 0;

    // set edges
    net.edges[0].from = 0;
    net.edges[0].to = 1;
    net.edges[0].num_filter_params = 2;
    net.edges[0].filter_param_start = param_idx;
    net.params[param_idx++] = 0.5; // mu
    net.params[param_idx++] = 0.3; // sigma
    net.edges[0].values = NULL;
    net.edges[0].log_probs = NULL;
    net.edges[0].num_messages = 0;

    // set nodes
    for (int n = 0; n < net.num_nodes; n++) {
        net.nodes[n].bias_index = param_idx++;
        net.nodes[n].weight_indices = NULL;
        net.nodes[n].incoming_edge_indices = NULL;
        net.nodes[n].values = NULL;
        net.nodes[n].log_probs = NULL;
    }
    // for node 0
    net.nodes[0].num_incoming = 0;
    // initial messages for node 0
    net.nodes[0].num_messages = 2;
    net.nodes[0].values = malloc(sizeof(double) * 2);
    net.nodes[0].log_probs = malloc(sizeof(double) * 2);
    net.nodes[0].values[0] = 0.0;
    net.nodes[0].log_probs[0] = log(0.5);
    net.nodes[0].values[1] = 1.0;
    net.nodes[0].log_probs[1] = log(0.5);

    // for node 1
    net.nodes[1].num_incoming = 1;
    net.nodes[1].incoming_edge_indices = malloc(sizeof(int) * 1);
    net.nodes[1].incoming_edge_indices[0] = 0;
    net.nodes[1].weight_indices = malloc(sizeof(int) * 1);
    net.nodes[1].weight_indices[0] = param_idx++;
    net.params[net.nodes[1].weight_indices[0]] = 1.0;
    net.params[net.nodes[1].bias_index] = 0.0;
    // initial for node 1
    net.nodes[1].num_messages = 1;
    net.nodes[1].values = malloc(sizeof(double) * 1);
    net.nodes[1].log_probs = malloc(sizeof(double) * 1);
    net.nodes[1].values[0] = 0.0;
    net.nodes[1].log_probs[0] = 0.0; // log(1.0)

    // run a few steps
    printf("Initial:\nNode 0:\n");
    print_node_messages(&net.nodes[0]);
    printf("Node 1:\n");
    print_node_messages(&net.nodes[1]);

    for (int step = 0; step < 2; step++) {
        edge_step(&net);
        node_step(&net);
        printf("\nAfter step %d:\nNode 0:\n", step + 1);
        print_node_messages(&net.nodes[0]);
        printf("Node 1:\n");
        print_node_messages(&net.nodes[1]);
    }

    // cleanup, omitted for brevity

    return 0;
}
