import React from 'react';
import { cn } from '@/lib/utils';

export type FilterOption = 'ALL' | 'AVAILABLE' | 'OCCUPIED' | 'UNCERTAIN' | 'REJECTED' | 'RECOMMENDED';
export type SortOption =
  | 'scoreDesc'
  | 'scoreAsc'
  | 'freqAsc'
  | 'freqDesc';

interface Props {
  filter: FilterOption;
  setFilter: (f: FilterOption) => void;
  sort: SortOption;
  setSort: (s: SortOption) => void;
}

export const ChannelFilterBar: React.FC<Props> = ({ filter, setFilter, sort, setSort }) => {
  return (
    <div className="flex flex-wrap gap-4 items-center mb-4">
      <label className="text-sm text-slate-300">
        Filter:
        <select
          className={cn('ml-2 p-1 rounded bg-slate-800 text-slate-200', 'focus:outline-none')}
          value={filter}
          onChange={(e) => setFilter(e.target.value as FilterOption)}
        >
          <option value="ALL">All</option>
          <option value="AVAILABLE">Available</option>
          <option value="OCCUPIED">Occupied</option>
          <option value="UNCERTAIN">Uncertain</option>
          <option value="REJECTED">Rejected</option>
          <option value="RECOMMENDED">Recommended</option>
        </select>
      </label>
      <label className="text-sm text-slate-300">
        Sort:
        <select
          className={cn('ml-2 p-1 rounded bg-slate-800 text-slate-200', 'focus:outline-none')}
          value={sort}
          onChange={(e) => setSort(e.target.value as SortOption)}
        >
          <option value="scoreDesc">Score ↓</option>
          <option value="scoreAsc">Score ↑</option>
          <option value="freqDesc">Frequency ↓</option>
          <option value="freqAsc">Frequency ↑</option>
        </select>
      </label>
    </div>
  );
};

