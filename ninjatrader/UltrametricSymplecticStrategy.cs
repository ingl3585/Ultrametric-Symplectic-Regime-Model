#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;

// HTTP Client for API calls
using System.Net.Http;
using System.Net.Http.Headers;

// JSON serialization
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it.
namespace NinjaTrader.NinjaScript.Strategies
{
	/// <summary>
	/// Ultrametric-Symplectic Regime Model Strategy for NinjaTrader 8
	///
	/// Calls external Python API on bar close to get trading signals.
	/// Supports 15-minute bars only (as specified in project requirements).
	///
	/// REQUIREMENTS:
	/// 1. Python FastAPI server must be running (python server/app.py)
	/// 2. Add references to strategy:
	///    - System.Net.Http
	///    - Newtonsoft.Json (download from NuGet if needed)
	/// 3. Configure API URL in strategy properties
	///
	/// USAGE:
	/// 1. Apply to 15-minute chart with Calculate.OnBarClose
	/// 2. Configure API endpoint (default: http://localhost:8000)
	/// 3. Set position size and risk parameters
	/// 4. Run in Sim101 or Playback101 for testing
	/// </summary>
	public class UltrametricSymplecticStrategy : Strategy
	{
		#region Variables
		// API Configuration
		private string apiUrl = "http://localhost:8000";
		private int barsToSend = 10;  // K parameter - must match Python model
		private HttpClient httpClient;

		// Position Management
		private int defaultQuantity = 1;
		private int currentPosition = 0;  // Track our position

		// Signal tracking
		private int lastSignalDirection = 0;
		private double lastSignalSizeFactor = 0.0;
		private bool enableTradeLogging = true;

		// Performance tracking
		private int totalSignals = 0;
		private int tradesExecuted = 0;
		#endregion

		#region OnStateChange
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Ultrametric-Symplectic trading strategy that calls external Python API";
				Name										= "UltrametricSymplecticStrategy";
				Calculate									= Calculate.OnBarClose;  // CRITICAL: Only on bar close
				EntriesPerDirection							= 1;
				EntryHandling								= EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy				= true;
				ExitOnSessionCloseSeconds					= 30;
				IsFillLimitOnTouch							= false;
				MaximumBarsLookBack							= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution							= OrderFillResolution.Standard;
				Slippage									= 0;
				StartBehavior								= StartBehavior.WaitUntilFlat;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 20;

				// User-configurable parameters
				ApiUrl					= "http://localhost:8000";
				BarsToSend				= 10;
				DefaultQuantity			= 1;
				EnableTradeLogging		= true;
			}
			else if (State == State.Configure)
			{
				// Add 15-minute data series (primary)
				// Note: The chart you apply this to should already be 15-minute
				// This is just defensive coding
			}
			else if (State == State.DataLoaded)
			{
				// Initialize HTTP client
				httpClient = new HttpClient();
				httpClient.BaseAddress = new Uri(apiUrl);
				httpClient.DefaultRequestHeaders.Accept.Clear();
				httpClient.DefaultRequestHeaders.Accept.Add(
					new MediaTypeWithQualityHeaderValue("application/json"));
				httpClient.Timeout = TimeSpan.FromSeconds(5);

				Print(string.Format("{0} initialized - API: {1}", Name, apiUrl));
			}
			else if (State == State.Terminated)
			{
				// Cleanup
				if (httpClient != null)
				{
					httpClient.Dispose();
					httpClient = null;
				}

				Print(string.Format("{0} terminated - Signals: {1}, Trades: {2}",
					Name, totalSignals, tradesExecuted));
			}
		}
		#endregion

		#region OnBarUpdate
		protected override void OnBarUpdate()
		{
			// Safety checks
			if (CurrentBar < BarsToSend)
				return;

			if (CurrentBar < BarsRequiredToTrade)
				return;

			// Only process on primary data series (15-minute bars)
			if (BarsInProgress != 0)
				return;

			try
			{
				// Get signal from Python API
				var signal = GetSignalFromAPI();

				if (signal != null)
				{
					totalSignals++;
					lastSignalDirection = signal.Direction;
					lastSignalSizeFactor = signal.SizeFactor;

					// Execute trade based on signal
					ExecuteTrade(signal);
				}
			}
			catch (Exception ex)
			{
				// Log error but don't crash
				Print(string.Format("ERROR in OnBarUpdate: {0}", ex.Message));
			}
		}
		#endregion

		#region API Communication

		/// <summary>
		/// Call Python API to get trading signal
		/// </summary>
		private TradingSignal GetSignalFromAPI()
		{
			try
			{
				// Build bar data array
				var bars = new List<BarData>();

				// Get last K bars
				int startBar = Math.Max(0, CurrentBar - BarsToSend + 1);
				for (int i = startBar; i <= CurrentBar; i++)
				{
					bars.Add(new BarData
					{
						Timestamp = Time[CurrentBar - i].ToString("yyyy-MM-ddTHH:mm:ssZ"),
						Open = Open[CurrentBar - i],
						High = High[CurrentBar - i],
						Low = Low[CurrentBar - i],
						Close = Close[CurrentBar - i],
						Volume = Volume[CurrentBar - i]
					});
				}

				// Build request payload
				var requestData = new
				{
					bars = bars,
					instrument = Instrument.FullName,
					account = new
					{
						account_id = Account.Name,
						cash_value = Account.CashValue,
						realized_pnl = Account.RealizedPnL,
						unrealized_pnl = Account.UnrealizedPnL,
						total_buying_power = Account.TotalBuyingPower,
						position_quantity = Position.Quantity,
						position_avg_price = Position.AveragePrice
					}
				};

				// Serialize to JSON
				string jsonContent = JsonConvert.SerializeObject(requestData);
				var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

				// Make synchronous HTTP POST
				// Note: async is problematic in NinjaScript, using .Result for simplicity
				var response = httpClient.PostAsync("/signal", content).Result;

				if (response.IsSuccessStatusCode)
				{
					string responseContent = response.Content.ReadAsStringAsync().Result;
					var signal = JsonConvert.DeserializeObject<TradingSignal>(responseContent);

					Print(string.Format("Signal received: Direction={0}, Size={1:F2}, Model={2}",
						signal.Direction, signal.SizeFactor, signal.ModelUsed));

					return signal;
				}
				else
				{
					Print(string.Format("API Error: {0} - {1}", response.StatusCode, response.ReasonPhrase));
					return null;
				}
			}
			catch (Exception ex)
			{
				Print(string.Format("ERROR calling API: {0}", ex.Message));
				return null;
			}
		}

		/// <summary>
		/// Log trade to Python API for monitoring
		/// </summary>
		private void LogTradeToAPI(string side, int quantity, double price, double pnl)
		{
			if (!enableTradeLogging)
				return;

			try
			{
				var tradeLog = new
				{
					timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ssZ"),
					instrument = Instrument.FullName,
					side = side,
					quantity = quantity,
					price = price,
					realized_pnl = pnl,
					strategy = Name
				};

				string jsonContent = JsonConvert.SerializeObject(tradeLog);
				var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

				// Fire and forget (don't wait for response)
				var _ = httpClient.PostAsync("/trade_log", content);
			}
			catch (Exception ex)
			{
				// Don't let logging errors affect trading
				Print(string.Format("WARNING: Could not log trade: {0}", ex.Message));
			}
		}

		#endregion

		#region Trade Execution

		/// <summary>
		/// Execute trade based on signal
		/// </summary>
		private void ExecuteTrade(TradingSignal signal)
		{
			// Calculate position size
			int targetQuantity = (int)(defaultQuantity * signal.SizeFactor);

			// Determine target position
			int targetPosition = signal.Direction * targetQuantity;

			// No change needed
			if (targetPosition == currentPosition)
				return;

			// Execute position change
			if (targetPosition > currentPosition)
			{
				// Need to go more long (or less short)
				int quantity = targetPosition - currentPosition;

				if (currentPosition < 0)
				{
					// Cover short first
					ExitShort();
					currentPosition = 0;
				}

				if (targetPosition > 0)
				{
					// Go long
					EnterLong(quantity, "Long");
					currentPosition = targetPosition;
					tradesExecuted++;
					LogTradeToAPI("Long", quantity, Close[0], 0.0);
				}
			}
			else if (targetPosition < currentPosition)
			{
				// Need to go more short (or less long)
				int quantity = currentPosition - targetPosition;

				if (currentPosition > 0)
				{
					// Exit long first
					ExitLong();
					currentPosition = 0;
				}

				if (targetPosition < 0)
				{
					// Go short
					EnterShort(Math.Abs(quantity), "Short");
					currentPosition = targetPosition;
					tradesExecuted++;
					LogTradeToAPI("Short", Math.Abs(quantity), Close[0], 0.0);
				}
			}

			// Draw signal on chart
			if (targetPosition != 0)
			{
				string tag = string.Format("Signal_{0}", CurrentBar);
				if (targetPosition > 0)
					Draw.ArrowUp(this, tag, true, 0, Low[0] - 2 * TickSize, Brushes.Green);
				else
					Draw.ArrowDown(this, tag, true, 0, High[0] + 2 * TickSize, Brushes.Red);
			}
		}

		#endregion

		#region OnExecutionUpdate
		protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
		{
			// Log fills to API
			if (execution.Order != null && execution.Order.OrderState == OrderState.Filled)
			{
				string side = execution.Order.IsLong ? "Long" : "Short";
				LogTradeToAPI(side, quantity, price, Account.RealizedPnL);
			}
		}
		#endregion

		#region Properties

		[NinjaScriptProperty]
		[Display(Name="API URL", Description="Python API endpoint", Order=1, GroupName="API Configuration")]
		public string ApiUrl
		{
			get { return apiUrl; }
			set { apiUrl = value; }
		}

		[NinjaScriptProperty]
		[Range(5, int.MaxValue)]
		[Display(Name="Bars To Send", Description="Number of bars to send to API (K parameter)", Order=2, GroupName="API Configuration")]
		public int BarsToSend
		{
			get { return barsToSend; }
			set { barsToSend = Math.Max(5, value); }
		}

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="Default Quantity", Description="Base position size", Order=3, GroupName="Position Management")]
		public int DefaultQuantity
		{
			get { return defaultQuantity; }
			set { defaultQuantity = Math.Max(1, value); }
		}

		[NinjaScriptProperty]
		[Display(Name="Enable Trade Logging", Description="Send trade logs to API", Order=4, GroupName="API Configuration")]
		public bool EnableTradeLogging
		{
			get { return enableTradeLogging; }
			set { enableTradeLogging = value; }
		}

		#endregion

		#region Helper Classes

		/// <summary>
		/// Bar data for API request
		/// </summary>
		private class BarData
		{
			[JsonProperty("timestamp")]
			public string Timestamp { get; set; }

			[JsonProperty("open")]
			public double Open { get; set; }

			[JsonProperty("high")]
			public double High { get; set; }

			[JsonProperty("low")]
			public double Low { get; set; }

			[JsonProperty("close")]
			public double Close { get; set; }

			[JsonProperty("volume")]
			public double Volume { get; set; }
		}

		/// <summary>
		/// Trading signal from API
		/// </summary>
		private class TradingSignal
		{
			[JsonProperty("direction")]
			public int Direction { get; set; }

			[JsonProperty("size_factor")]
			public double SizeFactor { get; set; }

			[JsonProperty("model_used")]
			public string ModelUsed { get; set; }

			[JsonProperty("timestamp")]
			public string Timestamp { get; set; }
		}

		#endregion
	}
}
