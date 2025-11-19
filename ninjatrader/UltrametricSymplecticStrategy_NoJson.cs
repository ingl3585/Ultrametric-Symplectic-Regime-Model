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
using System.Globalization;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it.
namespace NinjaTrader.NinjaScript.Strategies
{
	/// <summary>
	/// Ultrametric-Symplectic Regime Model Strategy for NinjaTrader 8
	///
	/// NO EXTERNAL DEPENDENCIES VERSION - No Newtonsoft.Json required
	///
	/// Calls external Python API on bar close to get trading signals.
	/// Supports 15-minute bars only (as specified in project requirements).
	///
	/// REQUIREMENTS:
	/// 1. Python FastAPI server must be running (python server/app.py)
	/// 2. Add reference to: System.Net.Http
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

		// State tracking
		private bool hasTransitionedToRealtime = false;
		#endregion

		#region OnStateChange
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Ultrametric-Symplectic trading strategy that calls external Python API (No JSON dependency)";
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

				// Only log termination if we actually traded or transitioned to real-time
				// This prevents spam during internal restarts on startup
				if (hasTransitionedToRealtime || totalSignals > 0 || tradesExecuted > 0)
				{
					Print(string.Format("{0} terminated - Signals: {1}, Trades: {2}",
						Name, totalSignals, tradesExecuted));
				}
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

			// IMPORTANT: Only call API during real-time execution, not historical processing
			// This prevents spamming the API with 100+ calls on strategy startup
			if (State != State.Realtime)
			{
				// Skip historical bars - model is stateless, no need to call API
				return;
			}

			// Log once when transitioning to real-time
			if (!hasTransitionedToRealtime)
			{
				hasTransitionedToRealtime = true;
				Print("*** READY TO TRADE - Historical processing complete, now monitoring live bars ***");
			}

			try
			{
				// Get signal from Python API
				var signal = GetSignalFromAPI();

				if (signal != null)
				{
					totalSignals++;
					lastSignalDirection = signal.direction;
					lastSignalSizeFactor = signal.size_factor;

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
				// Build JSON manually (no Newtonsoft dependency)
				StringBuilder jsonBuilder = new StringBuilder();
				jsonBuilder.Append("{");

				// Add bars array
				jsonBuilder.Append("\"bars\":[");
				int startBar = Math.Max(0, CurrentBar - BarsToSend + 1);
				for (int i = startBar; i <= CurrentBar; i++)
				{
					if (i > startBar)
						jsonBuilder.Append(",");

					int lookbackIndex = CurrentBar - i;
					jsonBuilder.AppendFormat(CultureInfo.InvariantCulture,
						"{{\"timestamp\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5}}}",
						Time[lookbackIndex].ToString("yyyy-MM-ddTHH:mm:ssZ"),
						Open[lookbackIndex],
						High[lookbackIndex],
						Low[lookbackIndex],
						Close[lookbackIndex],
						Volume[lookbackIndex]);
				}
				jsonBuilder.Append("],");

				// Add instrument
				jsonBuilder.AppendFormat("\"instrument\":\"{0}\",", EscapeJsonString(Instrument.FullName));

				// Add account info
				jsonBuilder.AppendFormat(CultureInfo.InvariantCulture,
					"\"account\":{{\"account_id\":\"{0}\",\"cash_value\":{1},\"realized_pnl\":{2},\"unrealized_pnl\":{3},\"total_buying_power\":{4},\"position_quantity\":{5},\"position_avg_price\":{6}}}",
					EscapeJsonString(Account.Name),
					Account.Get(AccountItem.CashValue, Currency.UsDollar),
					Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar),
					Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar),
					Account.Get(AccountItem.BuyingPower, Currency.UsDollar),
					Position.Quantity,
					Position.AveragePrice);

				jsonBuilder.Append("}");

				string jsonContent = jsonBuilder.ToString();
				var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

				// Make synchronous HTTP POST
				var response = httpClient.PostAsync("/signal", content).Result;

				if (response.IsSuccessStatusCode)
				{
					string responseContent = response.Content.ReadAsStringAsync().Result;
					var signal = ParseSignalResponse(responseContent);

					if (signal != null)
					{
						Print(string.Format("Signal received: Direction={0}, Size={1:F2}, Model={2}",
							signal.direction, signal.size_factor, signal.model_used));
					}

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
		/// Parse JSON response manually (no Newtonsoft dependency)
		/// </summary>
		private TradingSignal ParseSignalResponse(string json)
		{
			try
			{
				var signal = new TradingSignal();

				// Extract direction
				signal.direction = ExtractIntValue(json, "direction");

				// Extract size_factor
				signal.size_factor = ExtractDoubleValue(json, "size_factor");

				// Extract model_used
				signal.model_used = ExtractStringValue(json, "model_used");

				// Extract timestamp
				signal.timestamp = ExtractStringValue(json, "timestamp");

				return signal;
			}
			catch (Exception ex)
			{
				Print(string.Format("ERROR parsing signal response: {0}", ex.Message));
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
				// Build JSON manually
				string jsonContent = string.Format(CultureInfo.InvariantCulture,
					"{{\"timestamp\":\"{0}\",\"instrument\":\"{1}\",\"side\":\"{2}\",\"quantity\":{3},\"price\":{4},\"realized_pnl\":{5},\"strategy\":\"{6}\"}}",
					DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ssZ"),
					EscapeJsonString(Instrument.FullName),
					side,
					quantity,
					price,
					pnl,
					EscapeJsonString(Name));

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
			int targetQuantity = (int)(defaultQuantity * signal.size_factor);

			// Determine target position
			int targetPosition = signal.direction * targetQuantity;

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
				double realizedPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
				LogTradeToAPI(side, quantity, price, realizedPnL);
			}
		}
		#endregion

		#region JSON Helper Methods

		/// <summary>
		/// Escape string for JSON
		/// </summary>
		private string EscapeJsonString(string str)
		{
			if (string.IsNullOrEmpty(str))
				return str;

			return str.Replace("\\", "\\\\")
				.Replace("\"", "\\\"")
				.Replace("\n", "\\n")
				.Replace("\r", "\\r")
				.Replace("\t", "\\t");
		}

		/// <summary>
		/// Extract integer value from JSON string
		/// </summary>
		private int ExtractIntValue(string json, string key)
		{
			string pattern = string.Format("\"{0}\":", key);
			int startIndex = json.IndexOf(pattern);
			if (startIndex < 0)
				return 0;

			startIndex += pattern.Length;
			int endIndex = json.IndexOfAny(new char[] { ',', '}' }, startIndex);
			if (endIndex < 0)
				endIndex = json.Length;

			string valueStr = json.Substring(startIndex, endIndex - startIndex).Trim();
			int value;
			if (int.TryParse(valueStr, out value))
				return value;

			return 0;
		}

		/// <summary>
		/// Extract double value from JSON string
		/// </summary>
		private double ExtractDoubleValue(string json, string key)
		{
			string pattern = string.Format("\"{0}\":", key);
			int startIndex = json.IndexOf(pattern);
			if (startIndex < 0)
				return 0.0;

			startIndex += pattern.Length;
			int endIndex = json.IndexOfAny(new char[] { ',', '}' }, startIndex);
			if (endIndex < 0)
				endIndex = json.Length;

			string valueStr = json.Substring(startIndex, endIndex - startIndex).Trim();
			double value;
			if (double.TryParse(valueStr, NumberStyles.Any, CultureInfo.InvariantCulture, out value))
				return value;

			return 0.0;
		}

		/// <summary>
		/// Extract string value from JSON string
		/// </summary>
		private string ExtractStringValue(string json, string key)
		{
			string pattern = string.Format("\"{0}\":\"", key);
			int startIndex = json.IndexOf(pattern);
			if (startIndex < 0)
				return "";

			startIndex += pattern.Length;
			int endIndex = json.IndexOf("\"", startIndex);
			if (endIndex < 0)
				return "";

			return json.Substring(startIndex, endIndex - startIndex);
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
		/// Trading signal from API
		/// </summary>
		private class TradingSignal
		{
			public int direction { get; set; }
			public double size_factor { get; set; }
			public string model_used { get; set; }
			public string timestamp { get; set; }
		}

		#endregion
	}
}
