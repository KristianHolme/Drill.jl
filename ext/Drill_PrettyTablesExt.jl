module Drill_PrettyTablesExt

using Drill: Drill, RLCache, training_metric_rows
using PrettyTables: PrettyTables, pretty_table
import Drill: print_training_table

function print_training_table(cache::RLCache)
    rows = training_metric_rows(cache)
    isempty(rows) && return nothing
    names = first.(rows)
    values = last.(rows)
    pretty_table(
        hcat(names, values);
        column_labels = ["metric", "value"],
        alignment = [:l, :r],
    )
    return nothing
end

end
