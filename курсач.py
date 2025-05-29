import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.widgets import RadioButtons, Button
from matplotlib.patches import Rectangle
from datetime import datetime
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class MetroGraphApp:
    def __init__(self):
        # Инициализация данных
        self.df = pd.read_csv("metro_stations.csv")
        self.line_colors = {
            1: '#EF161E', 2: '#2DBE2C', 3: '#0078BE', 4: '#00BFFF', 5: '#8D5B2D',
            6: '#ED9121', 7: '#800080', 8: '#FFD702', 9: '#999999', 10: '#99CC00',
            11: '#82C0C0', 12: '#A1B3D4', 14: '#EF161E', 15: '#DE64A1', 16: '#03795F',
            17: '#27303F', 18: '#AC1753'
        }

        # Состояние приложения
        self.selected_stations = []
        self.text_box = None
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)
        self.passenger_type = "standard"

        # Создание графа
        self.G = self._create_metro_graph()
        self.pos = {node[0]: node[1]['pos'] for node in self.G.nodes(data=True)}

        # Границы графика
        self.base_xlim = (self.df.Lon.min() - 0.01, self.df.Lon.max() + 0.01)
        self.base_ylim = (self.df.Lat.min() - 0.01, self.df.Lat.max() + 0.01)

        # Инициализация интерфейса
        self._init_ui()

    def _create_metro_graph(self):
        """Создает граф метро из данных CSV"""
        G = nx.Graph()
        transfer_nodes = set()

        # Добавляем станции
        for _, row in self.df.iterrows():
            if row['Line'] == 13:
                continue

            G.add_node(row['Station_index'],
                       name=row['English_name'],
                       runame=row['Russian_name'],
                       line=row['Line'],
                       transfer = row['Transfers'],
                       pos=(row['Lon'], row['Lat']),
                       is_transfer=False)

        # Добавляем ребра (соседние станции на одной линии)
        for _, row in self.df.iterrows():
            if row['Line'] == 13:
                continue

            for neighbor in str(row['Line_Neighbors']).split():
                neighbor = int(neighbor)
                if neighbor in G.nodes:
                    G.add_edge(row['Station_index'], neighbor,
                               weight=2,
                               color=self.line_colors[row['Line']])

        # Добавляем пересадки (станции с одинаковыми именами)
        for name in self.df['English_name'].unique():
            same_name_stations = self.df[(self.df['English_name'] == name) &
                                         (self.df['Line'] != 13)]
            if len(same_name_stations) > 1:
                stations = same_name_stations['Station_index'].tolist()
                for i in range(len(stations)):
                    for j in range(i + 1, len(stations)):
                        u, v = stations[i], stations[j]
                        G.add_edge(u, v, weight=5, color='gray')
                        transfer_nodes.add(u)
                        transfer_nodes.add(v)
                        G.nodes[u]['is_transfer'] = True
                        G.nodes[v]['is_transfer'] = True

        return G

    def _init_ui(self):
        """Инициализирует пользовательский интерфейс"""
        self.fig, self.ax = plt.subplots(figsize=(50, 50), dpi=20)
        plt.subplots_adjust(bottom=0.3, right=0.8)

        # Радиокнопки для выбора типа пассажира
        self.rax = plt.axes([0.15, 0.05, 0.7, 0.15])
        self.radio = RadioButtons(
            self.rax,
            ('Стандартный пассажир (100% времени)',
             'Спешащий пассажир (-10% времени)',
             'Пассажир с ограниченными возможностями (+15% времени)'),
            active=0,

            activecolor='red'
        )


        # Увеличиваем размер элементов интерфейса
        for circle in self.radio.ax.findobj(match=plt.Circle):
            circle.set_radius(0.06)
        for label in self.radio.labels:
            label.set_fontsize(50)

        # Кнопка сброса
        self.reset_ax = plt.axes([0.82, 0.5, 0.15, 0.1])
        self.reset_button = Button(
            self.reset_ax,
            'Сбросить маршрут',
            color='lightgoldenrodyellow',
            hovercolor='0.975'
        )
        self.reset_button.label.set_fontsize(57)

        # Подключение обработчиков событий
        self.reset_button.on_clicked(self._reset_selection)
        self.radio.on_clicked(self._passenger_type_changed)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        # Первоначальная отрисовка
        self.ax.set_xlim(self.base_xlim[0], self.base_xlim[1])
        self.ax.set_ylim(self.base_ylim[0], self.base_ylim[1])
        self._draw_full_graph()
        plt.suptitle("Московское метро", fontsize=120, y=1)

    def _get_time_factor(self):
        """Возвращает коэффициент времени суток и описание периода"""
        now = datetime.now()
        current_hour = now.hour + now.minute / 60

        if 5.5 <= current_hour < 7:
            return 1.15, "Утро (5:30-7:00) +15%"
        elif 7 <= current_hour < 9:
            return 0.9, "Час пик (7:00-9:00) -10%"
        elif 9 <= current_hour < 17:
            return 1.0, "Обычное время (9:00-17:00)"
        elif 17 <= current_hour < 19:
            return 0.9, "Час пик (17:00-19:00) -10%"
        elif 19 <= current_hour < 24 or 0 <= current_hour < 1:
            return 1.0, "Обычное время (19:00-1:00)"
        else:
            return 1.2, "Ночь (1:00-5:30) +20%"

    def _calculate_adjusted_time(self, time):
        """Рассчитывает время с учетом типа пассажира и времени суток"""
        time_factor, _ = self._get_time_factor()

        if self.passenger_type == "hurrying":
            return time * 0.9 * time_factor
        elif self.passenger_type == "limited_mobility":
            return time * 1.15 * time_factor
        return time * time_factor

    def _reset_selection(self, event):
        """Обработчик сброса выбора"""
        self.selected_stations = []
        if self.text_box is not None:
            self.text_box.remove()
            self.text_box = None
        self._draw_full_graph()
        plt.draw()
        print("Выбор маршрута сброшен. Можете выбрать новый маршрут.")

    def _passenger_type_changed(self, label):
        """Обработчик изменения типа пассажира"""
        if label.startswith('Стандартный'):
            self.passenger_type = "standard"
        elif label.startswith('Спешащий'):
            self.passenger_type = "hurrying"
        elif label.startswith('Пассажир с ограниченными'):
            self.passenger_type = "limited_mobility"
        print(f"Выбран тип пассажира: {label}")

        if len(self.selected_stations) == 2:
            self._on_click(None)  # Пересчитываем маршрут

    def _on_click(self, event):
        """Обработчик клика мыши"""
        if event is not None and event.inaxes in [self.rax, self.reset_ax]:
            return

        if event is None or event.button == 1:  # Левая кнопка мыши
            if event is None or event.inaxes is not None:
                if event is not None:
                    distances = [(node, (event.xdata - self.pos[node][0]) ** 2 +
                                  (event.ydata - self.pos[node][1]) ** 2)
                                 for node in self.G.nodes()]
                    closest_node = min(distances, key=lambda x: x[1])[0]
                    node_name = self.G.nodes[closest_node]['name']
                    node_line = self.G.nodes[closest_node]['line']

                    if closest_node not in self.selected_stations:
                        self.selected_stations.append(closest_node)
                        print(f"Выбрана станция: {node_name} (линия {node_line}, ID: {closest_node})")

                if len(self.selected_stations) == 2:
                    self._calculate_and_draw_route()

        elif event is not None and event.button == 2:  # Средняя кнопка мыши
            self.is_dragging = True
            self.last_mouse_pos = (event.xdata, event.ydata)

    def _calculate_and_draw_route(self):
        """Рассчитывает и отображает маршрут между выбранными станциями"""
        start, end = self.selected_stations
        try:
            path = nx.shortest_path(self.G, source=start, target=end, weight='weight')
            path_time = sum(self.G[u][v]['weight'] for u, v in zip(path, path[1:]))

            time_factor, time_period = self._get_time_factor()
            adjusted_time = self._calculate_adjusted_time(path_time)

            route_text = f"Маршрут от {self.G.nodes[start]['runame']} до {self.G.nodes[end]['runame']}:\n"
            route_text += "→".join([self.G.nodes[n]['runame'] for n in path]) + "\n\n"
            route_text += f"Общее время: {adjusted_time:.1f} минут\n"
            route_text += f"Время суток: {time_period}\n"
            route_text += f"Тип пассажира: {'Стандартный' if self.passenger_type == 'standard' else 'Спешащий' if self.passenger_type == 'hurrying' else 'С ограниченными возможностями'}"

            self.ax.clear()

            # Рисуем только станции маршрута
            path_nodes = path
            path_edges = list(zip(path, path[1:]))

            # Все станции серым (неактивные)
            nx.draw_networkx_nodes(
                self.G, self.pos,
                nodelist=set(self.G.nodes()) - set(path_nodes),
                node_size=300, node_color='lightgray', alpha=0.5, ax=self.ax
            )

            # Станции маршрута
            nx.draw_networkx_nodes(
                self.G, self.pos,
                nodelist=path_nodes,
                node_size=600,
                node_color=[self.line_colors[self.G.nodes[n]['line']] for n in path_nodes],
                ax=self.ax
            )

            # Станции пересадок
            transfer_nodes_in_path = [n for n in path_nodes if self.G.nodes[n]['is_transfer']]
            nx.draw_networkx_nodes(
                self.G, self.pos,
                nodelist=transfer_nodes_in_path,
                node_size=800,
                node_color='yellow', alpha=0.7, ax=self.ax
            )

            # Начальная и конечная станции
            nx.draw_networkx_nodes(
                self.G, self.pos,
                nodelist=[start, end],
                node_size=800,
                node_color='red', ax=self.ax
            )

            # Ребра маршрута
            nx.draw_networkx_edges(
                self.G, self.pos,
                edgelist=path_edges,
                width=8,
                edge_color=[self.line_colors[self.G.nodes[u]['line']] for u, v in path_edges],
                ax=self.ax
            )

            # Подписи станций
            labels = {node: str(node) for node in path_nodes}
            nx.draw_networkx_labels(
                self.G, self.pos,
                labels=labels,
                font_size=10,
                font_weight='bold',
                ax=self.ax
            )

            # Легенда
            legend_elements = [
                Line2D([0], [0], color=color, lw=4, label=f'Линия {line}')
                for line, color in self.line_colors.items() if line != 13
            ]
            self.ax.legend(
                handles=legend_elements,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.25),
                fontsize=16,
                ncol=6
            )

            # Текст маршрута
            if self.text_box is not None:
                self.text_box.remove()

            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=4, alpha=0.9)
            self.text_box = self.ax.text(
                0.5, 1.05, route_text, transform=self.ax.transAxes,
                fontsize=35, color='black', ha='center', va='center', bbox=bbox_props
            )

            plt.draw()

        except nx.NetworkXNoPath:
            path_text = f"Нет маршрута от {self.G.nodes[start]['name']} до {self.G.nodes[end]['name']}"
            print(f"Маршрут не найден между {self.G.nodes[start]['name']} и {self.G.nodes[end]['name']}")

            if self.text_box is not None:
                self.text_box.remove()

            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="red", lw=2, alpha=0.8)
            self.text_box = self.ax.text(
                0.5, 1.05, path_text, transform=self.ax.transAxes,
                fontsize=24, color='red', ha='center', va='center', bbox=bbox_props
            )

            self._draw_full_graph()

        self.selected_stations = []

    def _on_release(self, event):
        """Обработчик отпускания кнопки мыши"""
        if event.button == 2:
            self.is_dragging = False

    def _on_motion(self, event):
        """Обработчик перемещения мыши"""
        if self.is_dragging and event.inaxes is not None:
            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]
            self.ax.set_xlim(self.ax.get_xlim()[0] - dx, self.ax.get_xlim()[1] - dx)
            self.ax.set_ylim(self.ax.get_ylim()[0] - dy, self.ax.get_ylim()[1] - dy)
            self.last_mouse_pos = (event.xdata, event.ydata)
            plt.draw()

    def _on_scroll(self, event):
        """Обработчик масштабирования"""
        if event.inaxes is not None:
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None:
                return

            zoom_factor = 1.2

            # Текущие границы
            x_left, x_right = self.ax.get_xlim()
            y_bottom, y_top = self.ax.get_ylim()

            # Новые границы
            if event.button == 'up':  # Приближение
                new_width = (x_right - x_left) / zoom_factor
                new_height = (y_top - y_bottom) / zoom_factor
            elif event.button == 'down':  # Отдаление
                new_width = (x_right - x_left) * zoom_factor
                new_height = (y_top - y_bottom) * zoom_factor
            else:
                return

            # Новые границы относительно позиции курсора
            new_x_left = xdata - (xdata - x_left) * (new_width / (x_right - x_left))
            new_x_right = xdata + (x_right - xdata) * (new_width / (x_right - x_left))
            new_y_bottom = ydata - (ydata - y_bottom) * (new_height / (y_top - y_bottom))
            new_y_top = ydata + (y_top - ydata) * (new_height / (y_top - y_bottom))

            self.ax.set_xlim(new_x_left, new_x_right)
            self.ax.set_ylim(new_y_bottom, new_y_top)

            plt.draw()

    def _draw_full_graph(self):
        """Отрисовывает полный граф метро"""
        self.ax.clear()

        # Рисуем граф
        nx.draw(
            self.G, self.pos,
            with_labels=True,
            node_color=[self.line_colors[self.G.nodes[n]['line']] for n in self.G.nodes()],
            edge_color=[self.G[u][v]['color'] for u, v in self.G.edges()],
            width=3,
            font_weight='bold',
            ax=self.ax,
            node_size=500,
            font_size=5,
        )

        # Легенда
        legend_elements = [
            Line2D([0], [0], color=color, lw=4, label=f'Линия {line}')
            for line, color in self.line_colors.items() if line != 13
        ]
        self.ax.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.25),
            fontsize=16,
            ncol=6
        )

        # Текущее время и коэффициент
        time_factor, time_period = self._get_time_factor()
        time_text = f"Текущее время: {datetime.now().strftime('%H:%M')} | {time_period}"
        self.ax.text(
            0.5, -0.15, time_text, transform=self.ax.transAxes,
            fontsize=80, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.7)
        )

        plt.draw()

    def run(self):
        """Запускает приложение"""
        plt.show()


# Создание и запуск приложения
if __name__ == "__main__":
    app = MetroGraphApp()
    app.run()
