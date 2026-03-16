/**
 * TimeSeriesChart Component - Plotly-based interactive charts
 * Supports: forecasts, anomaly detection, multi-model comparison, ensemble views
 */

const TimeSeriesChart = {
    name: 'TimeSeriesChart',

    props: {
        // Chart type: 'forecast', 'anomaly', 'multi_model', 'ensemble'
        chartType: {
            type: String,
            default: 'forecast'
        },
        // Skill result data from backend
        skillResult: {
            type: Object,
            required: true
        },
        // Historical data context (optional)
        historicalData: {
            type: Object,
            default: null
        },
        // Chart title
        title: {
            type: String,
            default: ''
        },
        // Session ID for fetching zoom data
        sessionId: {
            type: String,
            default: null
        }
    },

    template: `
        <div class="chart-wrapper">
            <div class="chart-header" v-if="title || hasMultipleViews || hasDownloadableData">
                <span class="chart-title">{{ title || defaultTitle }}</span>
                <div class="chart-header-actions">
                    <div class="chart-controls" v-if="hasMultipleViews">
                        <button 
                            v-for="view in availableViews" 
                            :key="view.id"
                            :class="['chart-view-btn', { active: currentView === view.id }]"
                            @click="currentView = view.id"
                        >
                            {{ view.label }}
                        </button>
                    </div>
                    
                    <!-- Download Button -->
                    <div class="download-container" v-if="hasDownloadableData">
                        <button class="download-btn" @click="toggleDownloadMenu" title="Download data">
                            <span class="download-icon">⬇</span>
                            <span class="download-text">Download</span>
                        </button>
                        
                        <!-- Download Dropdown Menu -->
                        <div v-if="showDownloadMenu" class="download-dropdown">
                            <!-- Data Type Selection -->
                            <div class="download-section">
                                <div class="download-section-title">DATA TYPE</div>
                                <div class="download-options-list">
                                    <div 
                                        v-for="option in downloadOptions" 
                                        :key="option.type"
                                        :class="['download-option', { selected: selectedDataType === option.type }]"
                                        @click="selectDataType(option.type)"
                                    >
                                        <span class="option-icon">{{ option.icon }}</span>
                                        <span class="option-label">{{ option.label }}</span>
                                        <span v-if="selectedDataType === option.type" class="option-check">✓</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Format Selection -->
                            <div class="download-section">
                                <div class="download-section-title">FORMAT</div>
                                <div class="format-toggle">
                                    <button 
                                        :class="['format-btn', { active: selectedFormat === 'csv' }]"
                                        @click.stop="selectedFormat = 'csv'"
                                    >CSV</button>
                                    <button 
                                        :class="['format-btn', { active: selectedFormat === 'json' }]"
                                        @click.stop="selectedFormat = 'json'"
                                    >JSON</button>
                                </div>
                            </div>
                            
                            <!-- Download Button -->
                            <div class="download-action">
                                <button 
                                    :class="['download-action-btn', { ready: isReadyToDownload }]" 
                                    @click="handleDownloadClick"
                                >
                                    ⬇ Download {{ selectedFormat.toUpperCase() }}
                                </button>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
            <div ref="chartContainer" class="chart-container"></div>
            <div class="chart-legend" v-if="showLegendSummary">
                <div v-for="item in legendItems" :key="item.name" class="legend-item">
                    <span class="legend-color" :style="{ background: item.color }"></span>
                    <span class="legend-label">{{ item.name }}</span>
                </div>
            </div>
        </div>
    `,

    data() {
        return {
            currentView: 'default',
            plotlyInstance: null,
            isZoomLoading: false,
            zoomedHistoricalData: null,  // Stores high-res data when zoomed
            currentZoomRange: null,      // Current visible range
            showDownloadMenu: false,     // Download dropdown visibility
            selectedFormat: 'csv',       // Default download format
            selectedDataType: null       // Selected data type for download
        };
    },


    computed: {
        defaultTitle() {
            const titles = {
                forecast: '📈 Forecast',
                anomaly: '🔍 Anomaly Detection',
                multi_model: '🔬 Multi-Model Comparison',
                ensemble: '📊 Ensemble Summary',
                overview: '📊 Time Series Overview',
                basic: '📊 Time Series'
            };
            return titles[this.chartType] || 'Time Series';
        },

        hasMultipleViews() {
            return this.chartType === 'multi_model' || this.chartType === 'ensemble';
        },

        availableViews() {
            if (this.chartType === 'multi_model') {
                return [
                    { id: 'all', label: 'All Models' },
                    { id: 'comparison', label: 'Comparison' },
                    { id: 'ensemble', label: 'Ensemble' }
                ];
            }
            if (this.chartType === 'ensemble') {
                return [
                    { id: 'summary', label: 'Summary' },
                    { id: 'spread', label: 'Spread' }
                ];
            }
            return [];
        },

        showLegendSummary() {
            return this.chartType === 'multi_model' && this.currentView === 'all';
        },

        legendItems() {
            const modelResults = this.skillResult?.model_results || this.chartData?.model_results || {};
            if (Object.keys(modelResults).length === 0) return [];
            
            const colors = this.getModelColors();
            return Object.keys(modelResults).map(name => ({
                name,
                color: colors[name] || '#666'
            }));
        },

        chartData() {
            return this.skillResult?.data || this.skillResult || {};
        },

        hasDownloadableData() {
            // Any chart with data can be downloaded
            const data = this.chartData;
            if (!data || Object.keys(data).length === 0) return false;
            
            // Check for various data types
            if (data.predictions || data.forecast) return true;
            if (data.anomalies || data.high_confidence_anomalies) return true;
            if (this.skillResult?.model_results) return true;
            if (data.chart_series || data.trend_data) return true;
            if (this.historicalData?.values) return true;
            
            return false;
        },

        downloadOptions() {
            const options = [];
            const data = this.chartData;
            
            // Always offer historical data if available
            if (this.historicalData?.values) {
                options.push({
                    type: 'historical',
                    label: 'Historical Data',
                    icon: '📊'
                });
            }
            
            // Forecast data
            if (data.predictions || data.forecast) {
                options.push({
                    type: 'forecast',
                    label: 'Forecast Results',
                    icon: '📈'
                });
            }
            
            // Anomaly data
            if (data.anomalies || data.high_confidence_anomalies || data.anomaly_indices) {
                options.push({
                    type: 'anomalies',
                    label: 'Detected Anomalies',
                    icon: '🔍'
                });
            }
            
            // NOTE: Multi-model comparison download removed per user request
            // User prefers to download the ensemble forecast via "Forecast Results"
            
            // Decomposition data
            if (data.chart_series || data.trend_data) {
                options.push({
                    type: 'decomposition',
                    label: 'Decomposition',
                    icon: '📉'
                });
            }
            
            // Simulation data
            if (data.chart_data && data.chart_data.original_values && data.chart_data.simulated_values) {
                options.push({
                    type: 'simulation',
                    label: 'Simulation Data',
                    icon: '🧪'
                });
            }
            
            // Full report (always available if we have skill result)
            if (this.skillResult?.skill_name) {
                options.push({
                    type: 'report',
                    label: 'Full Report (JSON)',
                    icon: '📋'
                });
            }
            
            return options;
        },

        isReadyToDownload() {
            // Ready if: only one option (auto-selected) OR user has selected a data type
            if (this.downloadOptions.length === 1) return true;
            return this.selectedDataType !== null;
        }
    },

    watch: {
        skillResult: {
            handler() {
                this.$nextTick(() => this.renderChart());
            },
            deep: true
        },
        currentView() {
            this.$nextTick(() => this.renderChart());
        }
    },

    mounted() {
        this.$nextTick(() => {
            this.renderChart();
            this.setupZoomListener();
        });
        window.addEventListener('resize', this.handleResize);
    },

    beforeUnmount() {
        window.removeEventListener('resize', this.handleResize);
        if (this.plotlyInstance) {
            Plotly.purge(this.$refs.chartContainer);
        }
    },

    methods: {
        setupZoomListener() {
            if (!this.$refs.chartContainer || !this.sessionId) return;
            
            // Listen for Plotly relayout events (zoom, pan, reset)
            this.$refs.chartContainer.on('plotly_relayout', (eventData) => {
                this.handlePlotlyRelayout(eventData);
            });
        },

        handlePlotlyRelayout(eventData) {
            // Check if this is a zoom event with x-axis range
            if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
                const startTime = eventData['xaxis.range[0]'];
                const endTime = eventData['xaxis.range[1]'];
                this.fetchDataForRange(startTime, endTime);
            } else if (eventData['xaxis.autorange'] === true) {
                // User double-clicked to reset zoom
                this.resetToOverview();
            }
        },

        async fetchDataForRange(startTime, endTime) {
            if (!this.sessionId || this.isZoomLoading) return;
            
            // Only fetch for overview/basic charts (not forecast results)
            if (this.chartType !== 'overview' && this.chartType !== 'basic') return;
            
            this.isZoomLoading = true;
            this.currentZoomRange = { start: startTime, end: endTime };
            
            try {
                const url = `/api/session/${this.sessionId}/chart-data?start_time=${encodeURIComponent(startTime)}&end_time=${encodeURIComponent(endTime)}&resolution=full&max_points=2000`;
                const response = await fetch(url);
                
                if (response.ok) {
                    const data = await response.json();
                    this.zoomedHistoricalData = data;
                    
                    // Update the chart with new high-resolution data
                    this.updateChartWithZoomedData(data);
                }
            } catch (e) {
                console.warn('Could not fetch zoomed data:', e);
            } finally {
                this.isZoomLoading = false;
            }
        },

        updateChartWithZoomedData(data) {
            if (!this.$refs.chartContainer) return;
            
            // Update trace data directly without full re-render
            const update = {
                x: [data.timestamps],
                y: [data.values]
            };
            
            Plotly.update(this.$refs.chartContainer, update, {}, [0]);
        },

        resetToOverview() {
            // Clear zoomed data and reset to original historical data
            this.zoomedHistoricalData = null;
            this.currentZoomRange = null;
            this.renderChart();
        },

        handleResize() {
            if (this.$refs.chartContainer && this.plotlyInstance) {
                Plotly.Plots.resize(this.$refs.chartContainer);
            }
        },

        /**
         * LTTB (Largest Triangle Three Buckets) downsampling algorithm.
         * Preserves visual appearance by keeping points that form largest triangles.
         */
        lttbDownsample(timestamps, values, targetPoints) {
            const n = values.length;
            if (targetPoints >= n || targetPoints < 3) {
                return { timestamps, values };
            }

            const sampledIndices = [0]; // Always keep first point
            const bucketSize = (n - 2) / (targetPoints - 2);
            let a = 0;

            for (let i = 0; i < targetPoints - 2; i++) {
                const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
                const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, n - 1);
                
                const nextBucketStart = Math.floor((i + 2) * bucketSize) + 1;
                const nextBucketEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, n);
                
                let avgX, avgY;
                if (nextBucketStart < n) {
                    avgX = (nextBucketStart + Math.min(nextBucketEnd, n)) / 2;
                    let sum = 0;
                    for (let k = nextBucketStart; k < nextBucketEnd; k++) sum += values[k];
                    avgY = sum / Math.max(1, nextBucketEnd - nextBucketStart);
                } else {
                    avgX = n - 1;
                    avgY = values[n - 1] || 0;
                }

                let maxArea = -1;
                let maxIdx = bucketStart;

                for (let j = bucketStart; j < bucketEnd; j++) {
                    const area = Math.abs(
                        (a - avgX) * (values[j] - values[a]) -
                        (a - j) * (avgY - values[a])
                    );
                    if (area > maxArea) {
                        maxArea = area;
                        maxIdx = j;
                    }
                }

                sampledIndices.push(maxIdx);
                a = maxIdx;
            }

            sampledIndices.push(n - 1); // Always keep last point

            return {
                timestamps: sampledIndices.map(i => timestamps[i]),
                values: sampledIndices.map(i => values[i])
            };
        },

        /**
         * Calculate target points for predictions to match historical data frequency.
         * This ensures predictions are downsampled to have the same visual density as historical data.
         */
        calculateFrequencyMatchedPoints(predTimestamps, predLength) {
            let targetPoints = Math.min(predLength, 500); // Default fallback
            
            if (this.historicalData?.timestamps && this.historicalData.timestamps.length > 1 && predTimestamps && predTimestamps.length > 1) {
                // Calculate time density of displayed historical data (ms per point)
                const histTimestamps = this.historicalData.timestamps;
                const histStart = new Date(histTimestamps[0]).getTime();
                const histEnd = new Date(histTimestamps[histTimestamps.length - 1]).getTime();
                const histTimeSpan = histEnd - histStart;
                const histPointCount = histTimestamps.length;
                
                if (histTimeSpan > 0 && histPointCount > 1) {
                    const msPerHistPoint = histTimeSpan / (histPointCount - 1);
                    
                    // Calculate time span of predictions
                    const predStart = new Date(predTimestamps[0]).getTime();
                    const predEnd = new Date(predTimestamps[predTimestamps.length - 1]).getTime();
                    const predTimeSpan = predEnd - predStart;
                    
                    if (predTimeSpan > 0 && msPerHistPoint > 0) {
                        // Calculate how many points we need to match the frequency
                        targetPoints = Math.max(2, Math.round(predTimeSpan / msPerHistPoint) + 1);
                        // Cap at reasonable limits
                        targetPoints = Math.min(targetPoints, predLength);
                        targetPoints = Math.min(targetPoints, 1000); // Hard cap for performance
                    }
                }
            }
            
            return targetPoints;
        },

        getModelColors() {
            const palette = [
                '#2563eb', '#dc2626', '#16a34a', '#ca8a04', 
                '#9333ea', '#0891b2', '#c026d3', '#ea580c'
            ];
            const modelResults = this.skillResult?.model_results || this.chartData?.model_results || {};
            const models = Object.keys(modelResults);
            const colors = {};
            models.forEach((name, idx) => {
                colors[name] = palette[idx % palette.length];
            });
            return colors;
        },

        getLayout() {
            return {
                autosize: true,
                height: 350,
                margin: { l: 50, r: 30, t: 30, b: 50 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 12, color: '#4a4a4a' },
                xaxis: {
                    showgrid: true,
                    gridcolor: '#e0e0e0',
                    linecolor: '#e0e0e0',
                    tickfont: { size: 11 }
                },
                yaxis: {
                    showgrid: true,
                    gridcolor: '#e0e0e0',
                    linecolor: '#e0e0e0',
                    tickfont: { size: 11 }
                },
                legend: {
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1
                },
                hovermode: 'x unified'
            };
        },

        getConfig() {
            return {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'timeseries_chart',
                    height: 600,
                    width: 1000,
                    scale: 2
                }
            };
        },

        renderChart() {
            if (!this.$refs.chartContainer) return;
            
            const data = this.chartData;
            if (!data || Object.keys(data).length === 0) return;

            let traces = [];

            switch (this.chartType) {
                case 'forecast':
                    traces = this.buildForecastTraces(data);
                    break;
                case 'anomaly':
                    traces = this.buildAnomalyTraces(data);
                    break;
                case 'multi_model':
                    traces = this.buildMultiModelTraces();
                    break;
                case 'ensemble':
                    traces = this.buildEnsembleTraces();
                    break;
                case 'overview':
                case 'basic':
                    traces = this.buildOverviewTraces(data);
                    break;
                case 'backtest':
                    traces = this.buildBacktestTraces(data);
                    break;
                case 'simulation':
                    traces = this.buildSimulationTraces(data);
                    break;
                default:
                    traces = this.buildBasicTraces(data);
            }

            if (traces.length === 0) return;

            Plotly.newPlot(
                this.$refs.chartContainer,
                traces,
                this.getLayout(),
                this.getConfig()
            );

            this.plotlyInstance = true;
        },

        buildForecastTraces(data) {
            const traces = [];
            let timestamps = data.timestamps || data.forecast_timestamps || 
                               this.generateTimestamps(data.predictions?.length || data.forecast?.length || 0);
            let predictions = data.predictions || data.forecast || [];

            // Downsample predictions to match historical data frequency
            const targetPoints = this.calculateFrequencyMatchedPoints(timestamps, predictions.length);
            if (predictions.length > targetPoints && targetPoints >= 2) {
                const downsampled = this.lttbDownsample(timestamps, predictions, targetPoints);
                timestamps = downsampled.timestamps;
                predictions = downsampled.values;
            }

            // Historical data if available
            if (this.historicalData?.values) {
                const histTimestamps = this.historicalData.timestamps || this.generateTimestamps(this.historicalData.values.length);
                traces.push({
                    x: histTimestamps,
                    y: this.historicalData.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Historical',
                    line: { color: '#6b7280', width: 2 }
                });

                // Add connection line from last historical point to first prediction
                if (predictions.length > 0 && this.historicalData.values.length > 0) {
                    const lastHistTimestamp = histTimestamps[histTimestamps.length - 1];
                    const lastHistValue = this.historicalData.values[this.historicalData.values.length - 1];
                    traces.push({
                        x: [lastHistTimestamp, timestamps[0]],
                        y: [lastHistValue, predictions[0]],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Connection',
                        line: { color: '#2563eb', width: 2, dash: 'dot' },
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                }
            }

            // Confidence band (upper/lower bounds) - also downsample if present
            if (data.lower_bound && data.upper_bound) {
                let lowerBound = data.lower_bound;
                let upperBound = data.upper_bound;
                let bandTimestamps = data.timestamps || this.generateTimestamps(lowerBound.length);
                
                if (lowerBound.length > targetPoints) {
                    const dsLower = this.lttbDownsample(bandTimestamps, lowerBound, targetPoints);
                    const dsUpper = this.lttbDownsample(bandTimestamps, upperBound, targetPoints);
                    lowerBound = dsLower.values;
                    upperBound = dsUpper.values;
                    bandTimestamps = dsLower.timestamps;
                }
                
                traces.push({
                    x: [...bandTimestamps, ...bandTimestamps.slice().reverse()],
                    y: [...upperBound, ...lowerBound.slice().reverse()],
                    fill: 'toself',
                    fillcolor: 'rgba(37, 99, 235, 0.15)',
                    line: { color: 'transparent' },
                    type: 'scatter',
                    name: 'Confidence Interval',
                    showlegend: true,
                    hoverinfo: 'skip'
                });
            }

            // Prediction line - clean line without markers
            if (predictions.length > 0) {
                traces.push({
                    x: timestamps,
                    y: predictions,
                    type: 'scatter',
                    mode: 'lines',
                    name: data.model || 'Forecast',
                    line: { color: '#2563eb', width: 2.5, shape: 'spline' },
                    connectgaps: true
                });
            }

            return traces;
        },


        buildAnomalyTraces(data) {
            const traces = [];
            
            // Get historical values and timestamps
            const values = this.historicalData?.values || data.values || [];
            const timestamps = this.historicalData?.timestamps || 
                               data.timestamps || 
                               this.generateTimestamps(values.length);

            // Main time series line
            if (values.length > 0) {
                traces.push({
                    x: timestamps,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Time Series',
                    line: { color: '#6b7280', width: 2 }
                });
            }

            // Handle new format: anomalies as array of objects with timestamp, value, score
            const anomalies = data.anomalies || [];
            const highConfidenceAnomalies = data.high_confidence_anomalies || [];
            const mediumConfidenceAnomalies = data.medium_confidence_anomalies || [];
            
            // Plot high confidence anomalies (red, large markers)
            if (highConfidenceAnomalies.length > 0) {
                const anomalyX = highConfidenceAnomalies.map(a => a.timestamp);
                const anomalyY = highConfidenceAnomalies.map(a => a.value);
                const hoverText = highConfidenceAnomalies.map(a => 
                    `High Confidence Anomaly<br>Value: ${a.value?.toFixed(2)}<br>Score: ${a.score || 'N/A'}<br>Detected by: ${a.detected_by || 'N/A'} models`
                );

                traces.push({
                    x: anomalyX,
                    y: anomalyY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'High Confidence',
                    marker: { 
                        size: 14, 
                        color: '#dc2626',
                        symbol: 'circle',
                        line: { color: '#fff', width: 2 }
                    },
                    text: hoverText,
                    hoverinfo: 'text'
                });
            }

            // Plot medium confidence anomalies (orange, medium markers)
            if (mediumConfidenceAnomalies.length > 0) {
                const anomalyX = mediumConfidenceAnomalies.map(a => a.timestamp);
                const anomalyY = mediumConfidenceAnomalies.map(a => a.value);
                const hoverText = mediumConfidenceAnomalies.map(a => 
                    `Medium Confidence Anomaly<br>Value: ${a.value?.toFixed(2)}<br>Score: ${a.score || 'N/A'}<br>Detected by: ${a.detected_by || 'N/A'} models`
                );

                traces.push({
                    x: anomalyX,
                    y: anomalyY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Medium Confidence',
                    marker: { 
                        size: 10, 
                        color: '#f59e0b',
                        symbol: 'diamond',
                        line: { color: '#fff', width: 1.5 }
                    },
                    text: hoverText,
                    hoverinfo: 'text'
                });
            }

            // Fallback: handle old format with anomaly_indices
            const anomalyIndices = data.anomaly_indices || [];
            const anomalyScores = data.anomaly_scores || [];
            
            if (anomalyIndices.length > 0 && values.length > 0) {
                const anomalyX = anomalyIndices.map(i => timestamps[i]).filter(t => t !== undefined);
                const anomalyY = anomalyIndices.map(i => values[i]).filter(v => v !== undefined);
                const hoverText = anomalyIndices.map((idx, i) => 
                    `Anomaly<br>Score: ${(anomalyScores[i] || 0).toFixed(2)}`
                );

                traces.push({
                    x: anomalyX,
                    y: anomalyY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Anomalies',
                    marker: { 
                        size: 12, 
                        color: '#dc2626',
                        symbol: 'circle',
                        line: { color: '#fff', width: 2 }
                    },
                    text: hoverText,
                    hoverinfo: 'text+x+y'
                });
            }

            // If we have anomalies list but no high/medium split, show all as general anomalies
            if (anomalies.length > 0 && highConfidenceAnomalies.length === 0 && mediumConfidenceAnomalies.length === 0) {
                const anomalyX = anomalies.map(a => a.timestamp);
                const anomalyY = anomalies.map(a => a.value);
                const hoverText = anomalies.map(a => 
                    `Anomaly<br>Value: ${a.value?.toFixed(2) || a.value}<br>Score: ${a.score || 'N/A'}`
                );

                traces.push({
                    x: anomalyX,
                    y: anomalyY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Anomalies',
                    marker: { 
                        size: 12, 
                        color: '#dc2626',
                        symbol: 'circle',
                        line: { color: '#fff', width: 2 }
                    },
                    text: hoverText,
                    hoverinfo: 'text'
                });
            }

            // Threshold line (if available from standard stats)
            if (data.threshold && values.length > 0) {
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const std = Math.sqrt(values.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / values.length);
                const thresholdUpper = mean + data.threshold * std;
                const thresholdLower = mean - data.threshold * std;

                traces.push({
                    x: [timestamps[0], timestamps[timestamps.length - 1]],
                    y: [thresholdUpper, thresholdUpper],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Upper Bound',
                    line: { color: '#dc2626', width: 1, dash: 'dash' },
                    hoverinfo: 'none'
                });

                traces.push({
                    x: [timestamps[0], timestamps[timestamps.length - 1]],
                    y: [thresholdLower, thresholdLower],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Lower Bound',
                    line: { color: '#dc2626', width: 1, dash: 'dash' },
                    hoverinfo: 'none'
                });
            }

            // Rule-based parsed thresholds (if parsed_rules metadata exists)
            if (data.metadata && data.metadata.parsed_rules && data.metadata.parsed_rules.conditions && values.length > 0) {
                data.metadata.parsed_rules.conditions.forEach((cond) => {
                    const drawLine = (yVal, name, color) => {
                        traces.push({
                            x: [timestamps[0], timestamps[timestamps.length - 1]],
                            y: [yVal, yVal],
                            type: 'scatter',
                            mode: 'lines',
                            name: name,
                            line: { color: color, width: 2, dash: 'dash' },
                            hoverinfo: 'name+y'
                        });
                    };

                    if (cond.type === 'hard_trigger_with_band') {
                        if (cond.trigger_value) {
                            drawLine(cond.trigger_value, 'Trigger Threshold', '#ca8a04');
                        }
                        if (cond.sustain_value) {
                            drawLine(cond.sustain_value, 'Sustain Threshold', '#16a34a');
                        }
                    } else if (cond.type === 'threshold' && cond.value !== undefined) {
                        drawLine(cond.value, `${cond.direction || ''} Threshold`, '#dc2626');
                    } else if (cond.type === 'range') {
                        if (cond.high !== undefined) drawLine(cond.high, 'Upper Bound', '#dc2626');
                        if (cond.low !== undefined) drawLine(cond.low, 'Lower Bound', '#dc2626');
                    } else if (cond.type === 'amplitude_band') {
                        const center = cond.center || 0;
                        const amp = cond.amplitude || 0;
                        drawLine(center + amp, 'Band Upper', '#16a34a');
                        drawLine(center - amp, 'Band Lower', '#16a34a');
                        drawLine(center, 'Band Center', '#94a3b8');
                    }
                });
            }

            return traces;
        },

        buildMultiModelTraces() {
            const traces = [];
            const modelResults = this.skillResult?.model_results || this.chartData?.model_results || {};
            const colors = this.getModelColors();
            const data = this.chartData;

            // Historical data
            if (this.historicalData?.values) {
                const histTimestamps = this.historicalData.timestamps || this.generateTimestamps(this.historicalData.values.length);
                traces.push({
                    x: histTimestamps,
                    y: this.historicalData.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Historical',
                    line: { color: '#9ca3af', width: 2 }
                });
            }

            // Get ensemble timestamps (from data or first model result)
            let ensembleTimestamps = data.timestamps;
            if (!ensembleTimestamps && Object.keys(modelResults).length > 0) {
                const firstModel = Object.values(modelResults)[0];
                ensembleTimestamps = firstModel?.data?.timestamps;
            }
            if (!ensembleTimestamps) {
                ensembleTimestamps = this.generateTimestamps(data.forecast?.length || 0);
            }

            // Ensemble mean trace (always visible by default)
            if (data.forecast && data.forecast.length > 0) {
                let ensembleForecast = data.forecast;
                let timestamps = ensembleTimestamps;

                // Downsample ensemble to match historical frequency
                const targetPoints = this.calculateFrequencyMatchedPoints(timestamps, ensembleForecast.length);
                if (ensembleForecast.length > targetPoints && targetPoints >= 2) {
                    const downsampled = this.lttbDownsample(timestamps, ensembleForecast, targetPoints);
                    timestamps = downsampled.timestamps;
                    ensembleForecast = downsampled.values;
                }

                // Uncertainty band (if available)
                if (data.uncertainty && data.uncertainty.length > 0) {
                    let uncertainty = data.uncertainty;
                    let bandTimestamps = ensembleTimestamps;
                    
                    // Compute similar downsampling for bands
                    if (uncertainty.length > targetPoints && targetPoints >= 2) {
                        const dsUncertainty = this.lttbDownsample(bandTimestamps, uncertainty, targetPoints);
                        uncertainty = dsUncertainty.values;
                        bandTimestamps = dsUncertainty.timestamps;
                    }
                    
                    const upperBound = ensembleForecast.map((v, i) => v + (uncertainty[i] || 0));
                    const lowerBound = ensembleForecast.map((v, i) => v - (uncertainty[i] || 0));

                    traces.push({
                        x: [...timestamps, ...timestamps.slice().reverse()],
                        y: [...upperBound, ...lowerBound.slice().reverse()],
                        fill: 'toself',
                        fillcolor: 'rgba(34, 197, 94, 0.15)',
                        line: { color: 'transparent' },
                        type: 'scatter',
                        name: 'Ensemble Interval',
                        showlegend: true,
                        hoverinfo: 'skip'
                    });
                }

                // Connection line from historical to ensemble
                if (this.historicalData?.values && this.historicalData.values.length > 0) {
                    const histTimestamps = this.historicalData.timestamps || this.generateTimestamps(this.historicalData.values.length);
                    const lastHistTimestamp = histTimestamps[histTimestamps.length - 1];
                    const lastHistValue = this.historicalData.values[this.historicalData.values.length - 1];
                    traces.push({
                        x: [lastHistTimestamp, timestamps[0]],
                        y: [lastHistValue, ensembleForecast[0]],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Connection',
                        line: { color: '#16a34a', width: 3, dash: 'dot' },
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                }

                // Ensemble mean line (prominent, always visible)
                traces.push({
                    x: timestamps,
                    y: ensembleForecast,
                    type: 'scatter',
                    mode: 'lines',
                    name: '🎯 Ensemble Mean',
                    line: { color: '#16a34a', width: 3 },
                    visible: true
                });
            }

            // Individual model forecasts (hidden by default, toggle via legend)
            if (this.currentView === 'default' || this.currentView === 'all' || this.currentView === 'comparison') {
                Object.entries(modelResults).forEach(([name, result]) => {
                    if (!result.success || !result.data?.predictions) return;
                    
                    let predictions = result.data.predictions;
                    let timestamps = result.data.timestamps || this.generateTimestamps(predictions.length);

                    // Downsample predictions to match historical frequency
                    const targetPoints = this.calculateFrequencyMatchedPoints(timestamps, predictions.length);
                    if (predictions.length > targetPoints && targetPoints >= 2) {
                        const downsampled = this.lttbDownsample(timestamps, predictions, targetPoints);
                        timestamps = downsampled.timestamps;
                        predictions = downsampled.values;
                    }

                    traces.push({
                        x: timestamps,
                        y: predictions,
                        type: 'scatter',
                        mode: 'lines',
                        name: name,
                        line: { color: colors[name], width: 2 },
                        visible: 'legendonly'  // Hidden by default, click legend to show
                    });
                });
            }

            if (this.currentView === 'ensemble') {
                traces.push(...this.buildEnsembleTraces());
            }

            return traces;
        },

        buildEnsembleTraces() {
            const traces = [];
            const aggregated = this.skillResult?.aggregated || this.chartData?.aggregated || {};

            // Historical data
            if (this.historicalData?.values) {
                traces.push({
                    x: this.historicalData.timestamps || this.generateTimestamps(this.historicalData.values.length),
                    y: this.historicalData.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Historical',
                    line: { color: '#9ca3af', width: 2 }
                });
            }

            const mean = aggregated.mean_prediction || [];
            const timestamps = this.generateTimestamps(mean.length);

            // Min/Max spread band
            if (aggregated.min_prediction && aggregated.max_prediction) {
                traces.push({
                    x: [...timestamps, ...timestamps.slice().reverse()],
                    y: [...aggregated.max_prediction, ...aggregated.min_prediction.slice().reverse()],
                    fill: 'toself',
                    fillcolor: 'rgba(37, 99, 235, 0.1)',
                    line: { color: 'transparent' },
                    type: 'scatter',
                    name: 'Model Spread',
                    showlegend: true,
                    hoverinfo: 'skip'
                });
            }

            // Mean prediction
            if (mean.length > 0) {
                traces.push({
                    x: timestamps,
                    y: mean,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Ensemble Mean',
                    line: { color: '#2563eb', width: 3 },
                    marker: { size: 6, color: '#2563eb' }
                });
            }

            // Median prediction
            if (aggregated.median_prediction?.length > 0) {
                traces.push({
                    x: timestamps,
                    y: aggregated.median_prediction,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Ensemble Median',
                    line: { color: '#16a34a', width: 2, dash: 'dot' }
                });
            }

            return traces;
        },

        buildSimulationTraces(data) {
            const traces = [];
            
            if (!data.chart_data) return traces;
            
            const timestamps = data.chart_data.timestamps;
            const originalValues = data.chart_data.original_values;
            const simulatedValues = data.chart_data.simulated_values;
            const modIndex = data.chart_data.modification_start_index || 0;
            
            if (!timestamps || !originalValues || !simulatedValues) return traces;
            
            // Trace 1: Original Data (Full Series)
            // Use a thinner, less prominent line for the part that overlaps with modification if needed, 
            // but user asked for "different color for the original and the amplified part".
            // Actually, usually "Current" is the baseline.
            
            traces.push({
                x: timestamps,
                y: originalValues,
                type: 'scatter',
                mode: 'lines',
                name: 'Baseline',
                line: { color: '#6b7280', width: 2 }
            });
            
            // Trace 2: Simulated Data (Only the modified part)
            if (modIndex < timestamps.length) {
                const simX = timestamps.slice(modIndex);
                const simY = simulatedValues.slice(modIndex);
                
                // Add one point before to connect the lines visually (if not at start)
                if (modIndex > 0) {
                    simX.unshift(timestamps[modIndex - 1]);
                    simY.unshift(simulatedValues[modIndex - 1]);
                }
                
                traces.push({
                    x: simX,
                    y: simY,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Simulated (Amplified)',
                    line: { color: '#ef4444', width: 2.5, dash: 'solid' } // Red for amplification/change
                });
                
                // Add fill between original and simulated if desired? 
                // Maybe overkill, but helps visualize delta. 
                // Let's stick to lines as requested: "two lines altohether if there is full series modification"
            }
            
            // Trace 3: Projection connection (optional, if we want to show the projections mentioned in tool result)
            // The tool returns `projected_value`. We could add a marker at horizon.
            // But let's stick to the historical modification visualization for now.
            
            return traces;
        },

        buildBacktestTraces(data) {
            const traces = [];
            
            // Context data (historical data used for prediction)
            const contextData = data.context_data || {};
            const contextTimestamps = contextData.timestamps || [];
            const contextValues = contextData.values || [];
            
            // Predictions
            const predictions = data.predictions || {};
            const predTimestamps = predictions.timestamps || [];
            const predValues = predictions.values || [];
            
            // Actuals (if comparison mode)
            const actuals = data.actuals || {};
            const actualTimestamps = actuals.timestamps || [];
            const actualValues = actuals.values || [];
            
            // Trace 1: Context data (historical used for context)
            if (contextValues.length > 0) {
                traces.push({
                    x: contextTimestamps,
                    y: contextValues,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Context (Training Data)',
                    line: { color: '#6b7280', width: 2 }
                });
            }
            
            // Trace 2: Predictions line
            if (predValues.length > 0) {
                // Connection from last context point to first prediction
                if (contextValues.length > 0) {
                    traces.push({
                        x: [contextTimestamps[contextTimestamps.length - 1], predTimestamps[0]],
                        y: [contextValues[contextValues.length - 1], predValues[0]],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Connection',
                        line: { color: '#2563eb', width: 2, dash: 'dot' },
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                }
                
                traces.push({
                    x: predTimestamps,
                    y: predValues,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Predicted',
                    line: { color: '#2563eb', width: 2.5, shape: 'spline' }
                });
            }
            
            // Trace 4: Actual values (if available, for comparison)
            if (actualValues.length > 0) {
                traces.push({
                    x: actualTimestamps,
                    y: actualValues,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Actual',
                    line: { color: '#16a34a', width: 2.5 },
                    marker: { size: 6, color: '#16a34a' }
                });
                
                // Trace 5: Error fill (difference between prediction and actual)
                if (predValues.length === actualValues.length && predTimestamps.length === actualTimestamps.length) {
                    traces.push({
                        x: [...predTimestamps, ...predTimestamps.slice().reverse()],
                        y: [...predValues, ...actualValues.slice().reverse()],
                        fill: 'toself',
                        fillcolor: 'rgba(239, 68, 68, 0.2)',
                        line: { color: 'transparent' },
                        type: 'scatter',
                        name: 'Prediction Error',
                        showlegend: true,
                        hoverinfo: 'skip'
                    });
                }
            }
            
            return traces;
        },

        buildBasicTraces(data) {
            const values = data.values || data.predictions || [];
            const timestamps = data.timestamps || this.generateTimestamps(values.length);

            return [{
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Data',
                line: { color: '#2563eb', width: 2 },
                marker: { size: 4 }
            }];
        },

        buildOverviewTraces(data) {
            const traces = [];
            
            // Use historicalData for overview chart
            const values = this.historicalData?.values || data.values || [];
            const timestamps = this.historicalData?.timestamps || data.timestamps || this.generateTimestamps(values.length);

            if (values.length === 0) return [];

            // Check for generic chart_series data (new format - supports multiple series)
            const chartSeries = data.chart_series || this.skillResult?.data?.chart_series;
            
            // Check for legacy trend_data format
            const trendData = data.trend_data || this.skillResult?.data?.trend_data;
            
            if (chartSeries && Object.keys(chartSeries).length > 0) {
                // Generic: Plot all series from chart_series
                // First, optionally add raw data as faded background
                traces.push({
                    x: timestamps,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Raw Data',
                    line: { 
                        color: '#9ca3af', 
                        width: 1
                    },
                    opacity: 0.3
                });

                // Plot each series from chart_series
                Object.entries(chartSeries).forEach(([key, series]) => {
                    if (series.values && series.values.length > 0) {
                        traces.push({
                            x: series.timestamps || this.generateTimestamps(series.values.length),
                            y: series.values,
                            type: 'scatter',
                            mode: 'lines',
                            name: series.name || key,
                            line: { 
                                color: series.color || this.getSeriesColor(key), 
                                width: 2.5,
                                shape: 'spline'
                            },
                            hovertemplate: series.description 
                                ? `<b>${series.name || key}</b><br>%{x}<br>Value: %{y:.2f}<extra>${series.description}</extra>`
                                : undefined
                        });
                    }
                });
            } else if (trendData && trendData.values && trendData.values.length > 0) {
                // Legacy: Handle trend_data format (from describe_series)
                // Plot raw data as faded background
                traces.push({
                    x: timestamps,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Raw Data',
                    line: { 
                        color: '#9ca3af', 
                        width: 1
                    },
                    opacity: 0.4
                });

                // Plot trend line as the main prominent line
                traces.push({
                    x: trendData.timestamps,
                    y: trendData.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: trendData.name || 'Trend',
                    line: { 
                        color: trendData.color || '#2563eb', 
                        width: 3,
                        shape: 'spline'
                    }
                });
            } else {
                // No special data - show regular time series
                traces.push({
                    x: timestamps,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: this.historicalData?.target_col || 'Value',
                    line: { 
                        color: '#2563eb', 
                        width: 2,
                        shape: 'spline'
                    },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(37, 99, 235, 0.1)'
                });
            }

            return traces;
        },

        // Get a consistent color for a series key
        getSeriesColor(key) {
            const colorMap = {
                'trend': '#2563eb',      // Blue
                'seasonal': '#16a34a',   // Green
                'residual': '#dc2626',   // Red
                'noise': '#dc2626',      // Red
                'pattern': '#9333ea',    // Purple
                'anomaly': '#f59e0b',    // Amber
                'forecast': '#0891b2',   // Cyan
                'default': '#6b7280'     // Gray
            };
            return colorMap[key.toLowerCase()] || colorMap['default'];
        },

        generateTimestamps(length) {
            // Generate simple indices if no timestamps provided
            return Array.from({ length }, (_, i) => i + 1);
        },

        // Download functionality
        toggleDownloadMenu() {
            this.showDownloadMenu = !this.showDownloadMenu;
            
            if (this.showDownloadMenu) {
                // Auto-select if only one option available
                if (this.downloadOptions.length === 1) {
                    this.selectedDataType = this.downloadOptions[0].type;
                } else if (!this.selectedDataType) {
                    // Reset selection when opening
                    this.selectedDataType = null;
                }
                
                // Close menu when clicking outside
                setTimeout(() => {
                    document.addEventListener('click', this.closeDownloadMenu);
                }, 0);
            }
        },

        closeDownloadMenu(event) {
            const container = this.$el.querySelector('.download-container');
            if (container && !container.contains(event.target)) {
                this.showDownloadMenu = false;
                document.removeEventListener('click', this.closeDownloadMenu);
            }
        },

        selectDataType(dataType) {
            this.selectedDataType = dataType;
            
            // Auto-select JSON for Full Report
            if (dataType === 'report') {
                this.selectedFormat = 'json';
            }
        },

        handleDownloadClick() {
            // If only one option, auto-select and download
            if (this.downloadOptions.length === 1) {
                this.downloadData(this.downloadOptions[0].type, this.selectedFormat);
                return;
            }
            
            // Validate that a data type is selected
            if (!this.selectedDataType) {
                alert('Please select a data type to download');
                return;
            }
            
            this.downloadData(this.selectedDataType, this.selectedFormat);
        },

        async downloadData(dataType, format) {
            this.showDownloadMenu = false;
            document.removeEventListener('click', this.closeDownloadMenu);
            
            if (!this.sessionId) {
                console.error('No session ID available for download');
                return;
            }
            
            try {
                // First, store the current skill result on the server
                await this.storeSkillResult();
                
                // Build download URL
                const params = new URLSearchParams({
                    data_type: dataType,
                    format: format
                });
                
                // Add optional parameters based on data type
                if (dataType === 'anomalies') {
                    params.append('include_flags', 'true');
                }
                if (dataType === 'forecast') {
                    params.append('include_confidence', 'true');
                }
                
                const url = `/api/session/${this.sessionId}/download?${params.toString()}`;
                
                // Trigger download
                const response = await fetch(url);
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Download failed');
                }
                
                // Get filename from header or generate one
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = `download_${Date.now()}.${format}`;
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename="(.+)"/);
                    if (match) filename = match[1];
                }
                
                // Create blob and download
                const blob = await response.blob();
                this.triggerDownload(blob, filename);
                
            } catch (error) {
                console.error('Download error:', error);
                alert('Download failed: ' + error.message);
            }
        },

        async storeSkillResult() {
            if (!this.sessionId || !this.skillResult) return;
            
            try {
                await fetch(`/api/session/${this.sessionId}/store-result`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        skill_result: this.skillResult
                    })
                });
            } catch (error) {
                console.warn('Could not store skill result:', error);
            }
        },

        triggerDownload(blob, filename) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }
};

if (typeof window !== 'undefined') {
    window.TimeSeriesChart = TimeSeriesChart;
}
