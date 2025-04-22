import asyncio
import json
import time
import argparse
import collections
from llm_benchmark import run_benchmark
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
import matplotlib.pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

async def run_all_benchmarks(llm_url, api_key, model, use_long_context, adaptive_mode=False):
    # 更细粒度的并发配置
    configurations = [
        {"num_requests": 10, "concurrency": 1, "output_tokens": 100},
        {"num_requests": 25, "concurrency": 5, "output_tokens": 100},
        {"num_requests": 50, "concurrency": 10, "output_tokens": 100},
        {"num_requests": 100, "concurrency": 20, "output_tokens": 100},
        {"num_requests": 150, "concurrency": 30, "output_tokens": 100},
        {"num_requests": 250, "concurrency": 50, "output_tokens": 100},
    ]

    all_results = []
    console = Console()

    # 自适应模式下的参数
    if adaptive_mode:
        min_success_rate = 80.0  # 最低可接受成功率
        max_concurrency = 500    # 最大尝试并发数
        current_concurrency = 1   # 起始并发数
        step_size = 5            # 初始步长
        
        console.print("[bold cyan]运行自适应并发探测模式...[/bold cyan]")
        
        with Progress(
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]探测最佳并发...", total=100)
            
            while current_concurrency <= max_concurrency:
                progress.update(task, description=f"[cyan]测试并发数 {current_concurrency}...", advance=5)
                
                # 根据并发数调整请求数，保持合理的测试时间
                test_requests = min(current_concurrency * 5, 500)
                
                try:
                    results = await run_benchmark(
                        test_requests, current_concurrency, 30, 100, 
                        llm_url, api_key, model, use_long_context
                    )
                    all_results.append(results)
                    
                    # 计算成功率
                    success_rate = (results['successful_requests'] / results['total_requests']) * 100
                    console.print(f"并发数 {current_concurrency}: 成功率 {success_rate:.1f}%, RPS {results['requests_per_second']:.2f}")
                    
                    # 如果成功率低于阈值，停止测试
                    if success_rate < min_success_rate:
                        console.print(f"[bold yellow]成功率低于 {min_success_rate}%，停止增加并发[/bold yellow]")
                        break
                    
                    # 动态调整步长，根据并发数范围使用不同的步长
                    if current_concurrency >= 100:
                        step_size = 30  # 高并发区间使用更大步长
                    elif current_concurrency >= 20:
                        step_size = 10  # 中等并发区间
                    
                    # 增加并发数继续测试
                    current_concurrency += step_size
                    
                except Exception as e:
                    console.print(f"[bold red]测试并发数 {current_concurrency} 时出错: {str(e)}[/bold red]")
                    break
                
                # 等待系统冷却
                await asyncio.sleep(5)
            
            progress.update(task, completed=100)
    else:
        # 常规模式，使用预定义配置
        for i, config in enumerate(configurations):
            console.print(f"[bold cyan]运行基准测试 {i+1}/{len(configurations)}: 并发数 {config['concurrency']}...[/bold cyan]")
            try:
                results = await run_benchmark(
                    config['num_requests'], config['concurrency'], 30, 
                    config['output_tokens'], llm_url, api_key, model, use_long_context
                )
                all_results.append(results)
                
                # 简化进度反馈，但增加更多有用信息
                success_rate = (results['successful_requests'] / results['total_requests']) * 100
                console.print(f"完成: RPS={results['requests_per_second']:.2f}, 成功率={success_rate:.1f}%, 平均延迟={results['latency']['average']:.3f}秒")
                
            except Exception as e:
                console.print(f"[bold red]测试并发数 {config['concurrency']} 时出错: {str(e)}[/bold red]")
            
            # 等待系统冷却
            await asyncio.sleep(5)

    return all_results

def analyze_results(all_results):
    """分析所有测试结果并生成汇总报告"""
    summary = []
    total_tokens = 0
    total_time = 0
    
    # 简化调试信息
    print(f"分析 {len(all_results)} 个测试结果")
    
    for result in all_results:
        try:
            # 简化调试输出
            concurrency = result.get('concurrency', 0)
            
            concurrency = result.get('concurrency', 0)
            rps = result.get('requests_per_second', 0)
            
            # 更健壮的数据获取，提供默认值
            latency_data = result.get('latency', {})
            tps_data = result.get('tokens_per_second', {})
            ttft_data = result.get('time_to_first_token', {})
            
            avg_latency = latency_data.get('average', 0) if isinstance(latency_data, dict) else 0
            p99_latency = latency_data.get('p99', 0) if isinstance(latency_data, dict) else 0
            avg_tps = tps_data.get('average', 0) if isinstance(tps_data, dict) else 0
            avg_ttft = ttft_data.get('average', 0) if isinstance(ttft_data, dict) else 0
            
            total_requests = max(result.get('total_requests', 1), 1)  # 避免除以零
            success_rate = (result.get('successful_requests', 0) / total_requests) * 100
            
            # 确保所有值都是有效的数字
            invalid_values = []
            for name, value in [
                ("concurrency", concurrency), 
                ("rps", rps), 
                ("avg_latency", avg_latency), 
                ("p99_latency", p99_latency), 
                ("avg_tps", avg_tps), 
                ("avg_ttft", avg_ttft)
            ]:
                if value is None or not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    invalid_values.append(name)
            
            if invalid_values:
                print(f"警告: 并发数 {concurrency} 的测试结果包含无效数据: {', '.join(invalid_values)}，已跳过")
                continue

            summary.append([
                concurrency,
                f"{rps:.2f}",
                f"{avg_latency:.3f}",
                f"{p99_latency:.3f}",
                f"{avg_tps:.2f}",
                f"{avg_ttft:.3f}",
                f"{success_rate:.1f}%"
            ])
            
            total_tokens += result.get('total_output_tokens', 0) or 0
            total_time += result.get('total_time', 0) or 0
            
        except Exception as e:
            print(f"警告: 处理并发数 {result.get('concurrency', 'unknown')} 的结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not summary:
        print("错误: 没有有效的测试结果数据")
        return [], 0, 0

    print(f"成功分析 {len(summary)} 个有效测试结果")
    return summary, total_tokens, total_time

def print_summary(all_results, model_name, use_long_context):
    """打印测试结果汇总"""
    summary, total_tokens, total_time = analyze_results(all_results)
    
    if not summary:
        print("没有可用的测试结果数据进行展示")
        return
    
    console = Console(width=100)  # 设置固定宽度
    
    # 创建标题面板
    title = Text("性能测试汇总报告", style="bold")
    console.print(Panel(title, width=60))
    
    # 打印基本信息
    basic_info = Table(show_header=False, width=60)
    basic_info.add_column("名称", style="cyan", width=20)
    basic_info.add_column("值", style="green", width=40)
    
    basic_info.add_row("模型", model_name)
    basic_info.add_row("长文本模式", "是" if use_long_context else "否")
    basic_info.add_row("总生成Token数", f"{total_tokens:,}")
    basic_info.add_row("总测试时间", f"{total_time:.2f} 秒")
    basic_info.add_row("平均Token生成速率", f"{total_tokens/total_time:.2f} tokens/sec")
    
    console.print("\n基本信息:")
    console.print(basic_info)
    
    # 创建详细性能指标表格
    table = Table(
        title="详细性能指标",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
        width=100,  # 设置表格总宽度
        pad_edge=False,  # 减少边缘填充
        min_width=80,    # 最小宽度
    )
    
    # 添加列（设置固定列宽）
    table.add_column("并发数", justify="right", style="cyan", width=8)
    table.add_column("RPS", justify="right", width=8)
    table.add_column("平均延迟(秒)", justify="right", width=12)
    table.add_column("P99延迟(秒)", justify="right", width=12)
    table.add_column("平均TPS", justify="right", width=10)
    table.add_column("首Token延迟", justify="right", width=12)
    table.add_column("成功率", justify="right", style="green", width=8)
    
    # 添加数据行
    for row in summary:
        try:
            # 根据成功率设置行样式
            success_rate = float(row[6].rstrip('%'))
            row_style = "green" if success_rate >= 95 else "yellow" if success_rate >= 80 else "red"
            
            table.add_row(
                str(row[0]),                    # 并发数
                f"{float(row[1]):.2f}",        # RPS
                f"{float(row[2]):.3f}",        # 平均延迟
                f"{float(row[3]):.3f}",        # P99延迟
                f"{float(row[4]):.2f}",        # 平均TPS
                f"{float(row[5]):.3f}",        # 首Token延迟
                row[6],                        # 成功率
                style=row_style
            )
        except ValueError as e:
            console.print(f"警告: 处理行数据时出错: {str(e)}", style="bold red")
            continue
    
    console.print("\n")
    console.print(table)
    
    # 错误类型统计
    error_stats = collections.Counter()
    error_samples = {}
    
    for result in all_results:
        if 'error_statistics' in result:
            # 合并错误计数
            for error_type, count in result['error_statistics']['count'].items():
                error_stats[error_type] += count
            
            # 收集错误样本
            for error_type, samples in result['error_statistics']['samples'].items():
                if error_type not in error_samples:
                    error_samples[error_type] = []
                error_samples[error_type].extend(samples[:2])  # 每种类型最多取2个样本
    
    if error_stats:
        console.print("\n[bold red]错误统计:[/bold red]")
        error_table = Table(show_header=True, width=80)
        error_table.add_column("错误类型", style="red")
        error_table.add_column("次数", justify="right")
        error_table.add_column("百分比", justify="right")
        
        total_errors = sum(error_stats.values())
        for error_type, count in error_stats.most_common():
            percentage = (count / total_errors) * 100 if total_errors > 0 else 0
            error_table.add_row(
                error_type,
                str(count),
                f"{percentage:.1f}%"
            )
        
        console.print(error_table)
        
        # 错误样本展示
        if error_samples:
            console.print("\n[bold yellow]错误样本:[/bold yellow]")
            for error_type, samples in error_samples.items():
                console.print(f"[cyan]{error_type}[/cyan] 类型错误样本:")
                for i, sample in enumerate(samples[:3]):  # 最多显示3个样本
                    console.print(f"  {i+1}. {sample}", style="yellow")
    
    # 生成性能可视化图表
    try:
        # 提取数据
        concurrencies = [int(row[0]) for row in summary]
        rps_values = [float(row[1]) for row in summary]
        latencies = [float(row[2]) for row in summary]
        success_rates = [float(row[6].rstrip('%')) for row in summary]
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 1. RPS vs 并发数
        plt.subplot(2, 2, 1)
        plt.plot(concurrencies, rps_values, 'o-', color='blue')
        plt.title('RPS vs 并发数')
        plt.xlabel('并发数')
        plt.ylabel('每秒请求数 (RPS)')
        plt.grid(True)
        
        # 2. 延迟 vs 并发数
        plt.subplot(2, 2, 2)
        plt.plot(concurrencies, latencies, 'o-', color='red')
        plt.title('平均延迟 vs 并发数')
        plt.xlabel('并发数')
        plt.ylabel('平均延迟 (秒)')
        plt.grid(True)
        
        # 3. 成功率 vs 并发数
        plt.subplot(2, 2, 3)
        plt.plot(concurrencies, success_rates, 'o-', color='green')
        plt.title('成功率 vs 并发数')
        plt.xlabel('并发数')
        plt.ylabel('成功率 (%)')
        plt.ylim(0, 105)  # 设置y轴范围为0-105%
        plt.grid(True)
        
        # 4. 延迟分布 (最后一次测试)
        if all_results and 'latency' in all_results[-1]:
            plt.subplot(2, 2, 4)
            
            # 提取最后一次测试的延迟数据点
            last_result = all_results[-1]
            latency_values = [
                last_result['latency']['p50'],
                last_result['latency']['p95'],
                last_result['latency']['p99'],
            ]
            percentiles = ['P50', 'P95', 'P99']
            
            plt.bar(percentiles, latency_values, color=['green', 'orange', 'red'])
            plt.title(f'延迟分布 (并发数={last_result["concurrency"]})')
            plt.ylabel('延迟 (秒)')
            plt.grid(True, axis='y')
        
        # 保存图表
        plt.tight_layout()
        try:
            img_filename = 'benchmark_results.png'
            # 如果文件已存在，先删除它
            import os
            if os.path.exists(img_filename):
                os.remove(img_filename)
            
            plt.savefig(img_filename)
            console.print(f"\n[bold green]性能图表已保存至 {img_filename} (已覆盖)[/bold green]")
        except Exception as e:
            console.print(f"\n[bold red]保存性能图表时出错: {str(e)}[/bold red]")
        
    except Exception as e:
        console.print(f"\n[bold red]生成性能图表时出错: {str(e)}[/bold red]")
    
    # 计算和显示最佳性能配置
    try:
        best_rps_idx = np.argmax([float(row[1]) if row[1] != "N/A" else -1 for row in summary])
        best_latency_idx = np.argmin([float(row[2]) if row[2] != "N/A" else float('inf') for row in summary])
        
        perf_info = Table(title="性能最佳配置", show_header=False, box=None, width=60)
        perf_info.add_column("指标", style="cyan", width=15)
        perf_info.add_column("值", style="green", width=45)
        
        perf_info.add_row(
            "最高 RPS",
            f"并发数 {summary[best_rps_idx][0]} ({summary[best_rps_idx][1]} req/sec)"
        )
        perf_info.add_row(
            "最低延迟",
            f"并发数 {summary[best_latency_idx][0]} ({summary[best_latency_idx][2]} 秒)"
        )
        
        console.print("\n")
        console.print(perf_info)
        
        # 性能建议
        recommendations = []
        if best_rps_idx == len(summary) - 1:
            recommendations.append("系统似乎还未达到性能瓶颈，可以尝试更高的并发数")
        elif best_rps_idx == 0:
            recommendations.append("建议尝试降低并发数，当前负载可能过高")
        else:
            recommendations.append(f"最佳并发数范围在 {summary[best_rps_idx][0]} 附近")
        
        success_rate = float(summary[-1][6][:-1])
        if success_rate < 95:
            recommendations.append("在高并发时成功率偏低，建议检查系统资源或降低并发数")
        
        # 错误类型建议
        if 'timeout' in error_stats and error_stats['timeout'] > 0:
            recommendations.append("出现超时错误，可能需要增加请求超时时间或优化服务响应速度")
        if 'rate_limit' in error_stats and error_stats['rate_limit'] > 0:
            recommendations.append("出现速率限制错误，建议降低并发数或联系API提供方提高限制")
        if 'network_error' in error_stats and error_stats['network_error'] > 0:
            recommendations.append("出现网络错误，请检查网络连接稳定性")
        
        recommend_text = Text("\n性能建议:", style="bold cyan")
        console.print(recommend_text)
        for rec in recommendations:
            console.print(f"• {rec}", style="yellow")
            
    except Exception as e:
        console.print(f"警告: 生成性能分析时出错: {str(e)}", style="bold red")

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks with various configurations")
    parser.add_argument("--llm_url", type=str, required=True, help="URL of the LLM server")
    parser.add_argument("--api_key", type=str, required=False, default="default", help="API key for LLM server")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    parser.add_argument("--model", type=str, default="deepseek-r1", 
                       help="Model name to use for inference (default: deepseek-r1)")
    parser.add_argument("--adaptive", action="store_true", help="Run in adaptive mode")
    args = parser.parse_args()

    all_results = asyncio.run(run_all_benchmarks(args.llm_url, args.api_key, args.model, args.use_long_context, args.adaptive))

    # 保存详细结果到文件
    try:
        json_filename = 'benchmark_results.json'
        with open(json_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"详细测试结果已保存至 {json_filename} (已覆盖)")
    except Exception as e:
        print(f"保存JSON结果时出错: {str(e)}")
    
    # 打印汇总报告
    print_summary(all_results, args.model, args.use_long_context)

if __name__ == "__main__":
    main()