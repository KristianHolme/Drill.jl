module Drill_PrettyTablesExt

using Drill: Drill, RLCache, table_output, training_metric_rows
using PrettyTables: PrettyTables, pretty_table
import Drill: print_training_table

function print_training_table(cache::RLCache)
    rows = training_metric_rows(cache)
    isempty(rows) && return nothing
    names = first.(rows)
    values = last.(rows)
    pretty_table(
        table_output(cache.verbosity), hcat(names, values);
        column_labels = ["metric", "value"],
        alignment = [:l, :r],
    )
    return nothing
end

end
