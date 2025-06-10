import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


#_____________________TASK 1_____________________________________



# Визначаємо шляхи до файлів вихідних даних та чтаємо дані з визначених CSV-файлів

STOPS_FILE = 'transport_stops.csv'
ROUTE_SEGMENTS_FILE = 'route_segments.csv'
TRANSFERS_FILE = 'transfer.csv'
ROUTES_FILE = 'routes.csv'

try:
    stops_df = pd.read_csv(STOPS_FILE, sep=',')
    route_segments_df = pd.read_csv(ROUTE_SEGMENTS_FILE, sep=',')
    transfers_df = pd.read_csv(TRANSFERS_FILE, sep=',')
    routes_df = pd.read_csv(ROUTES_FILE, sep=',')

except FileNotFoundError as e:
    print(f"Помилка: Файл не знайдено - {e}. Перевірте шляхи до файлів.")
    exit()
except pd.errors.EmptyDataError:
    print("Помилка: Один з файлів порожній.")
    exit()
except Exception as e:
    print(f"Сталася помилка при читанні CSV: {e}")
    print("Перевірте, чи коректно вказані роздільники (sep=) та назви стовпців у ваших CSV-файлах.")
    exit()

# Побудова графа

G = nx.MultiGraph()

# Додавання вузлів (зупинок)
for index, row in stops_df.iterrows():
    G.add_node(row['stop_id'],
               name=row['stop_name'],
               transport=row['transport'],
               transfer_ability=row['transfer_ability'])

# Додавання ребер для відрізків маршрутів
# Отримуємо тип транспорту для маршруту з route_df
# Для цього об'єднаємо route_segments_df з routes_df

for index, row in route_segments_df.iterrows():
    
    route_info = routes_df[routes_df['route_id'] == row['route_id']].iloc[0]
    transport_type = route_info['transport']
    route_name = route_info['route_name']

    G.add_edge(row['from_stop'], row['to_stop'],
               weight=row['time'], 
               type='route_segment',
               route_segment_id=row['route_segment'],
               route_id=row['route_id'],
               route_name=route_name,
               transport_type=transport_type,
               order=row['order'])

# Додавання ребер для пересадок
for index, row in transfers_df.iterrows():
   
    if G.has_node(row['from_stop_id']) and G.has_node(row['to_stop_id']):
        G.add_edge(row['from_stop_id'], row['to_stop_id'],
                   weight=row['time'], 
                   type='transfer',
                   transfer_id=row['transfer_id'])
    else:
        print(f"Попередження: Не знайдено зупинку {row['from_stop_id']} або {row['to_stop_id']} для пересадки {row['transfer_id']}. Пропущено.")

print(f"\nГраф побудовано: {G.number_of_nodes()} вузлів, {G.number_of_edges()} ребер.")

# Візуалізація графа (для відносно реального відображення треба додати координати зупинок - не здійснювалось, застосовано автогенерацію розташування)

plt.figure(figsize=(15, 10))

pos = nx.spring_layout(G, k=0.5, iterations=50)

# Налаштування кольорів для вузлів за типом транспорту
node_colors = []

color_map = {
    'трамвай': 'red',
    'тролейбус': 'green',
    'mixed': 'purple',
    'unknown': 'darkgray'
}

for node_id in G.nodes():
    transport_type = G.nodes[node_id].get('transport', 'unknown').lower()
    node_colors.append(color_map.get(transport_type, color_map['unknown']))


# Налаштування кольорів та стилів для ребер
edge_colors = []
edge_styles = []
edge_widths = []
edge_labels = {} 
for u, v, key, data in G.edges(keys=True, data=True): 
    if data['type'] == 'route_segment':
        edge_colors.append(color_map.get(data['transport_type'].lower(), 'lightgrey'))
        edge_styles.append('solid')
        edge_widths.append(1.5)
    elif data['type'] == 'transfer':
        edge_colors.append('black') 
        edge_styles.append('dashed')
        edge_widths.append(1)
        

nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=100,
                       alpha=0.9)

nx.draw_networkx_edges(G, pos,
                       edge_color=edge_colors,
                       style=edge_styles,
                       width=edge_widths,
                       alpha=0.7)

node_labels = {node_id: G.nodes[node_id]['name'] for node_id in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')

plt.title("Граф громадського транспорту (трамвай, тролейбус) міста Вінниця", size=16)
plt.axis('off')
plt.show()

# Аналіз основних характеристик графа
print("\n Аналіз Характеристик Графа ")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"Кількість вузлів (зупинок): {num_nodes}")
print(f"Кількість ребер (відрізків маршрутів та пересадок): {num_edges}")

# Ступінь вершин (кількість зв'язків для кожної зупинки)

print("\nСтупінь Вершин")
degrees = dict(G.degree())
#out_degrees = dict(G.out_degree())

# Максимальний ступінь
max_in_degree = max(degrees.values())
nodes_with_max_in_degree = [node for node, degree in degrees.items() if degree == max_in_degree]
print(f"Максимальний ступінь: {max_in_degree} (зупинки: {[G.nodes[n]['name'] for n in nodes_with_max_in_degree]})")

# Мінімальний ступінь
min_degree = min(degrees.values())
nodes_with_min_degree = [node for node, degree in degrees.items() if degree == min_degree]
if len(nodes_with_min_degree) > 5:
    print(f"Мінімальний вхідний ступінь: {min_degree} (в тому числі, зупинки: {[G.nodes[n]['name'] for n in nodes_with_min_degree[:5]]}, ...)")
else:
    print(f"Мінімальний вхідний ступінь: {min_degree} (зупинки: {[G.nodes[n]['name'] for n in nodes_with_min_degree]})")







#_________________TASK 2________________________________




start_node = 's022'
end_node = 's111'

# Пошук шляху за допомогою DFS

def find_path_dfs(graph, start, end):

    if not graph.has_node(start) or not graph.has_node(end):
        return None, "Одна або обидві зупинки не існують у графі."

    visited = set()
    stack = [(start, [start])]

    while stack:
        current_node, path = stack.pop()

        if current_node == end:
            return path, None

        if current_node not in visited:
            visited.add(current_node)

            for neighbor in reversed(list(graph.neighbors(current_node))):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None, "Шлях не знайдено."


dfs_path, dfs_error = find_path_dfs(G, start_node, end_node)
if dfs_path:
    print("\nШлях, знайдений DFS:")
    print(" -> ".join([G.nodes[node]['name'] for node in dfs_path]))
    print(f"Кількість вузлів у шляху (довжина): {len(dfs_path)}")
    
    dfs_total_time = 0
    for i in range(len(dfs_path) - 1):
        u = dfs_path[i]
        v = dfs_path[i+1]
        
        edges_uv = G.get_edge_data(u, v)
        if edges_uv:
            min_edge_key = min(edges_uv, key=lambda k: edges_uv[k]['weight'])
            dfs_total_time += edges_uv[min_edge_key]['weight']
    print(f"Загальний час для шляху DFS: {dfs_total_time} хвилин.")
else:
    print(f"\nDFS: {dfs_error}")



# Пошук шляху за допомогою BFS

def find_path_bfs(graph, start, end):

    if not graph.has_node(start) or not graph.has_node(end):
        return None, "Одна або обидві зупинки не існують у графі."

    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current_node, path = queue.popleft()

        if current_node == end:
            return path, None

        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, "Шлях не знайдено."


bfs_path, bfs_error = find_path_bfs(G, start_node, end_node)
if bfs_path:
    print("\nШлях, знайдений BFS:")
    print(" -> ".join([G.nodes[node]['name'] for node in bfs_path]))
    print(f"Кількість вузлів у шляху (довжина): {len(bfs_path)}")
    
    bfs_total_time = 0
    for i in range(len(bfs_path) - 1):
        u = bfs_path[i]
        v = bfs_path[i+1]
        edges_uv = G.get_edge_data(u, v)
        if edges_uv:
            min_edge_key = min(edges_uv, key=lambda k: edges_uv[k]['weight'])
            bfs_total_time += edges_uv[min_edge_key]['weight']
    print(f"Загальний час для шляху BFS: {bfs_total_time} хвилин.")
else:
    print(f"\nBFS: {bfs_error}")




#_________________________TASK 3_____________________________________-

# Пошук найкоротшого шляху між визначеними вершинами .

try:
    
    if G.has_node(start_node) and G.has_node(end_node):
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
        
        print(f"\nНайкоротший шлях від {G.nodes[start_node]['name']} ({start_node}) до {G.nodes[end_node]['name']} ({end_node}):")
        print(f"Шлях: {path}")
        print(f"Загальний час у дорозі: {path_length} хвилин")
        
        print("\nДеталі шляху:")
        current_time = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]

            edges_between_nodes = G.get_edge_data(u, v)
            if edges_between_nodes:
                chosen_edge_key = min(edges_between_nodes, key=lambda x: edges_between_nodes[x]['weight'])
                edge_data = edges_between_nodes[chosen_edge_key]
                
                current_time += edge_data['weight']
                
                if edge_data['type'] == 'route_segment':
                    print(f"  {G.nodes[u]['name']} ({u}) --({edge_data['transport_type']} {edge_data['route_name']} | {edge_data['weight']} хв)--> {G.nodes[v]['name']} ({v}). Сумарний час: {current_time} хв.")
                elif edge_data['type'] == 'transfer':
                    print(f"  Пересадка з {G.nodes[u]['name']} ({u}) на {G.nodes[v]['name']} ({v}) --({edge_data['weight']} хв)-->. Сумарний час: {current_time} хв.")
            else:
                print(f"  Не знайдено ребра між {G.nodes[u]['name']} ({u}) та {G.nodes[v]['name']} ({v})")
    else:
        print(f"\nПомилка: Початкова зупинка '{start_node}' або кінцева зупинка '{end_node}' не знайдена в графі.")

except nx.NetworkXNoPath:
    print(f"\nНемає шляху від {G.nodes[start_node]['name']} ({start_node}) до {G.nodes[end_node]['name']} ({end_node}).")
except Exception as e:
    print(f"\nСталася помилка при пошуку найкоротшого шляху: {e}")



# Пошук найкоротших шляхів між усіма парами вершин за допомогою вбудованої реалізації алгоритма Дійкстри

# all_pairs_shortest_paths_lengths = {}
# try:
#     for source_node, distances in nx.all_pairs_dijkstra_path_length(G, weight='weight'):
#         all_pairs_shortest_paths_lengths[source_node] = distances
    
#     print("\nПриклади найкоротших відстаней між деякими парами зупинок (час у хвилинах):")
    
#     nodes_to_display = list(G.nodes())
    
#     for source in nodes_to_display:
#         for target in nodes_to_display:
#             if source != target and target in all_pairs_shortest_paths_lengths[source]:
#                 length = all_pairs_shortest_paths_lengths[source][target]
#                 print(f"Від {G.nodes[source]['name']} ({source}) до {G.nodes[target]['name']} ({target}): {length:.2f} хв")
#             elif source == target:
#                 print(f"Від {G.nodes[source]['name']} ({source}) до {G.nodes[target]['name']} ({target}): 0.00 хв (сама до себе)")
#             else:
#                 print(f"Від {G.nodes[source]['name']} ({source}) до {G.nodes[target]['name']} ({target}): Шлях не знайдено.")

#     distance_matrix = pd.DataFrame(all_pairs_shortest_paths_lengths).fillna(float('inf'))
#     print("\nМатриці відстаней (Час у хвилинах, Inf = немає шляху):")
#     print(distance_matrix.loc[nodes_to_display, nodes_to_display])


# except nx.NetworkXNoPath:
#     print("\nПомилка: Не всі вузли є доступними один від одного.")
# except Exception as e:
#     print(f"\nСталася помилка при обчисленні шляхів між усіма парами: {e}")


# Пошук найкоротших шляхів кастомним алгоритмом Дійкстри


def dijkstra(graph, start):
    
    distances = {vertex: float('infinity') for vertex in graph.nodes()}
    distances[start] = 0

    unvisited = set(graph.nodes())

    while unvisited:
       
        min_distance = float('infinity')
        current_vertex = None
        for vertex in unvisited:
            if distances[vertex] < min_distance:
                min_distance = distances[vertex]
                current_vertex = vertex

        if current_vertex is None or distances[current_vertex] == float('infinity'):
            break

        unvisited.remove(current_vertex)
        neighbors_weights = {}
        for u, v, key, data in graph.edges(current_vertex, data=True, keys=True):
            weight = data['weight'] 
            if v not in neighbors_weights or weight < neighbors_weights[v]:
                neighbors_weights[v] = weight

        for neighbor, weight in neighbors_weights.items():
            distance = distances[current_vertex] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance

    return distances

print("\nНайкоротші шляхи між усіма парами вершин")

all_pairs_shortest_paths_length_custom = {}

nodes_to_display = list(G.nodes())

for start_node_id in nodes_to_display:
    all_pairs_shortest_paths_length_custom[start_node_id] = dijkstra(G, start_node_id)


distance_matrix_custom = pd.DataFrame(all_pairs_shortest_paths_length_custom).fillna(float('inf'))
print("\nМатриці відстаней (Час у хвилинах, Inf = немає шляху):")
print(distance_matrix_custom.loc[nodes_to_display, nodes_to_display])
